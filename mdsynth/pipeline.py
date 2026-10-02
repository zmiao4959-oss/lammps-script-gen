"""
MDSynthPipeline — the main orchestrator.

Coordinates all 7 modules:
1. ScientificIntentExtractor (NL → Intent Spec)
2. MDTypedIRPlanner (Intent Spec → MD Typed IR)
3. PhysicsValidator (Validate IR)
4. ConstrainedRepairEngine (Auto-repair IR if needed)
5. LAMMPSBackendCompiler (IR → LAMMPS script)
6. SandboxPreflightRunner (Test-run script)
7. EvidencePackageBuilder (Assemble output package)
"""

from __future__ import annotations

from typing import Optional

from mdsynth.config import MDSynthConfig, get_config
from mdsynth.utils.logging import get_logger, configure_logging

from mdsynth.llm.base import LLMBackend, MockLLMBackend
from mdsynth.intent.extractor import ScientificIntentExtractor, IntentExtractionError
from mdsynth.intent.spec import ScientificIntentSpec
from mdsynth.ir.planner import MDTypedIRPlanner, IRPlanningError
from mdsynth.ir.md_ir import MDTypedIR
from mdsynth.validator.engine import PhysicsValidator
from mdsynth.validator.diagnostic import Diagnostic, Severity
from mdsynth.repair.engine import ConstrainedRepairEngine
from mdsynth.backend.compiler import LAMMPSBackendCompiler, CompilerError
from mdsynth.sandbox.runner import SandboxPreflightRunner
from mdsynth.sandbox.report import PreflightReport
from mdsynth.evidence.builder import EvidencePackageBuilder, MDSynthOutput
from mdsynth.llm_direct.generator import LLMDirectGenerator, DirectGenerationError
from mdsynth.llm_direct.rag import LAMMPSRAGClient

logger = get_logger(__name__)


class RepairExhaustedError(Exception):
    """Raised when repair rounds are exhausted without resolving errors."""
    pass


class MDSynthPipeline:
    """
    Main orchestrator for the MDSynth system.

    Usage:
        from mdsynth import MDSynthPipeline
        from mdsynth.llm.base import MockLLMBackend

        pipeline = MDSynthPipeline(llm_backend=MockLLMBackend())
        output = pipeline.run("铜在300K下NPT平衡200ps")
        pipeline.save(output, "./output/")
    """

    def __init__(
        self,
        llm_backend: LLMBackend | None = None,
        config: MDSynthConfig | None = None,
    ):
        self.config = config or get_config()

        # Initialize LLM backend
        if llm_backend is not None:
            self.llm = llm_backend
        else:
            self.llm = MockLLMBackend()

        # Initialize all modules
        self.intent_extractor = ScientificIntentExtractor(self.llm)
        self.ir_planner = MDTypedIRPlanner(self.llm)
        self.validator = PhysicsValidator()
        self.repair = ConstrainedRepairEngine(
            max_auto_rounds=self.config.max_repair_rounds
        )
        self.compiler = LAMMPSBackendCompiler()
        self.sandbox = SandboxPreflightRunner(
            lmp_executable=self.config.lammps_executable,
            potentials_dir=self.config.lammps_potentials_dir,
        )
        self.evidence_builder = EvidencePackageBuilder()

        # Configure logging
        configure_logging()

    def run(self, user_request: str) -> MDSynthOutput:
        """
        Execute the full pipeline on a user request.

        Args:
            user_request: Natural language MD research goal

        Returns:
            MDSynthOutput with script, evidence, and reports

        Raises:
            RepairExhaustedError: If repair rounds are exhausted
            IntentExtractionError: If intent extraction fails
            IRPlanningError: If IR planning fails
            CompilerError: If compilation fails
        """
        logger.info("pipeline_started", user_request=user_request)

        # ============================================================
        # Step 1: NL → Intent Spec
        # ============================================================
        logger.info("step_1_intent_extraction")
        try:
            intent_spec = self.intent_extractor.extract(user_request)
            intent_spec.original_user_text = user_request
        except IntentExtractionError:
            logger.error("intent_extraction_failed")
            raise

        logger.info(
            "intent_extracted",
            task_type=intent_spec.task_type,
            confidence=intent_spec.extraction_confidence.value,
            template_match=intent_spec.template_match,
        )

        # ============================================================
        # Branch: template vs. direct LLM generation
        # ============================================================
        if not intent_spec.template_match:
            logger.info("branch_direct_generation")
            return self._run_direct_generation(user_request, intent_spec)

        # ============================================================
        # Step 2: Intent Spec → MD Typed IR
        # ============================================================
        logger.info("step_2_ir_planning")
        try:
            md_ir = self.ir_planner.plan(intent_spec)
        except IRPlanningError:
            logger.error("ir_planning_failed")
            raise

        logger.info("ir_planned", num_stages=len(md_ir.stages))

        # ============================================================
        # Step 3-4: Validate IR → Repair loop
        # ============================================================
        logger.info("step_3_4_validation_repair_loop")
        diagnostics: list[Diagnostic] = []

        for round_num in range(self.config.max_repair_rounds):
            diagnostics = self.validator.validate(md_ir)
            errors = self.validator.get_errors(diagnostics)
            warnings = self.validator.get_warnings(diagnostics)

            logger.info(
                "validation_round",
                round=round_num + 1,
                errors=len(errors),
                warnings=len(warnings),
            )

            if not errors:
                break

            # Generate and apply repairs
            repair_actions = self.repair.generate_actions(md_ir, errors)
            md_ir = self.repair.apply_actions(md_ir, repair_actions)

            # Check for blocked repairs
            blocked = [a for a in repair_actions if a.action_type == "blocked"]
            if blocked:
                logger.warning("repair_blocked", blocked=[b.id for b in blocked])
                # Continue anyway — blocked actions are documented
        else:
            # Max rounds exceeded
            remaining_errors = self.validator.get_errors(
                self.validator.validate(md_ir)
            )
            if remaining_errors:
                raise RepairExhaustedError(
                    f"Repair exhausted after {self.config.max_repair_rounds} rounds. "
                    f"Remaining errors: {len(remaining_errors)}"
                )

        # Collect final diagnostics
        if not diagnostics:
            diagnostics = self.validator.validate(md_ir)

        # ============================================================
        # Step 5: Compile to LAMMPS script
        # ============================================================
        logger.info("step_5_compilation")
        try:
            lammps_script, lowered_ir = self.compiler.compile(md_ir)
        except CompilerError:
            logger.error("compilation_failed")
            raise

        logger.info("compilation_complete", script_lines=len(lammps_script.split("\n")))

        # ============================================================
        # Step 6: Sandbox preflight
        # ============================================================
        logger.info("step_6_sandbox_preflight")
        if self.config.sandbox_enabled:
            preflight_report = self.sandbox.run(lammps_script, md_ir)
        else:
            preflight_report = PreflightReport(passed=True, script_compiles=True)
            preflight_report.warning_messages.append("Sandbox preflight disabled by config")

        logger.info("preflight_complete", passed=preflight_report.passed)

        # ============================================================
        # Step 7: Build evidence package
        # ============================================================
        logger.info("step_7_evidence_package")
        output = self.evidence_builder.build(
            intent_spec=intent_spec,
            md_ir=md_ir,
            diagnostics=diagnostics,
            lammps_script=lammps_script,
            preflight_report=preflight_report,
            repair_history=self.repair.get_history(),
        )

        logger.info("pipeline_complete", success=output.success)
        return output

    def save(self, output: MDSynthOutput, directory: str | None = None) -> None:
        """
        Save the output package to disk.

        Args:
            output: The pipeline output
            directory: Target directory (uses config default if None)
        """
        from pathlib import Path

        target = Path(directory) if directory else self.config.output_directory
        self.evidence_builder.write_to_directory(output, target)
        logger.info("output_saved", directory=str(target))

    # ================================================================
    # Direct LLM Generation Path (non-template)
    # ================================================================

    def _run_direct_generation(
        self,
        user_request: str,
        intent_spec: ScientificIntentSpec,
    ) -> MDSynthOutput:
        """
        Fallback path: LLM generates LAMMPS script directly, validate with run-0.

        1. LLM generates complete LAMMPS script
        2. Replace run → run 0, execute via local LAMMPS binary
        3. If error → feed back to LLM → regenerate (max N rounds)
        4. Return script with original run values + repair history
        """
        rag_client = (
            LAMMPSRAGClient(
                self.config.rag_api_url,
                timeout_sec=self.config.rag_timeout_sec,
            )
            if self.config.rag_api_url
            else None
        )
        direct_gen = LLMDirectGenerator(
            llm_backend=self.llm,
            lmp_executable=self.config.lammps_executable,
            max_rounds=self.config.max_direct_gen_rounds,
            potentials_dir=self.config.lammps_potentials_dir,
            rag_client=rag_client,
            rag_required=self.config.rag_required,
            rag_script_limit=self.config.rag_script_limit,
            rag_command_limit=self.config.rag_command_limit,
        )

        try:
            lammps_script, repair_history = direct_gen.generate(
                user_request, intent_spec,
                work_dir=str(self.config.output_directory),
            )
        except DirectGenerationError:
            logger.error("direct_generation_failed")
            raise

        # Check if all rounds failed
        all_passed = any(h["passed"] for h in repair_history) if repair_history else False
        errors = [h["error"] for h in repair_history if not h["passed"]]

        # Build a simple preflight report from the repair history
        preflight_report = PreflightReport(
            passed=all_passed,
            script_compiles=all_passed,
        )
        if errors:
            preflight_report.error_messages = errors[-1:] if not all_passed else []
        if not all_passed:
            preflight_report.warning_messages.append(
                f"Direct generation: {len(repair_history)} round(s), "
                f"all failed — script may not be runnable"
            )

        logger.info(
            "direct_generation_complete",
            rounds=len(repair_history),
            passed=all_passed,
        )

        # Build output (no IR, no validator diagnostics for direct path)
        return self.evidence_builder.build(
            intent_spec=intent_spec,
            md_ir=None,
            diagnostics=[],
            lammps_script=lammps_script,
            preflight_report=preflight_report,
            repair_history=repair_history,
        )
