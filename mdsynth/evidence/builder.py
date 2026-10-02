"""
EvidencePackageBuilder — assembles the final output package.

Produces:
- LAMMPS input script (in.main.lammps)
- MD Typed IR serialization (md_ir.yaml)
- Intent spec serialization (intent_spec.yaml)
- Validation report (validation_report.json)
- Preflight report (preflight_report.json)
- Provenance manifest (provenance.lock)
- Assumptions document (assumptions.md)
- README (README.md)
- Semantic diff / repair history (semantic_diff.md)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union, TYPE_CHECKING

from mdsynth.utils.serialization import to_json, to_yaml, to_json_file, to_yaml_file
from mdsynth.evidence.manifest import ProvenanceManifest, ForceFieldProvenance
from mdsynth.evidence.report_templates import (
    build_readme,
    build_assumptions,
    build_semantic_diff,
)

if TYPE_CHECKING:
    from mdsynth.intent.spec import ScientificIntentSpec
    from mdsynth.ir.md_ir import MDTypedIR
    from mdsynth.validator.diagnostic import Diagnostic
    from mdsynth.sandbox.report import PreflightReport
    from mdsynth.repair.engine import RepairAction


@dataclass
class MDSynthOutput:
    """Complete output of the MDSynth pipeline."""
    # Core outputs
    lammps_script: str = ""
    md_ir_yaml: str = ""
    intent_spec_yaml: str = ""

    # Reports
    validation_report_json: str = ""
    preflight_report_json: str = ""
    provenance_lock_yaml: str = ""

    # Documents
    readme_md: str = ""
    assumptions_md: str = ""
    semantic_diff_md: str = ""
    repair_history_md: str = ""  # direct-gen repair trajectory

    # Structured data (for programmatic use)
    manifest: Optional[ProvenanceManifest] = None

    # Status
    success: bool = False
    error_message: str = ""


class EvidencePackageBuilder:
    """
    Assembles the complete output package from all pipeline artifacts.

    Collects Intent, IR, validation results, preflight results, and
    repair history into a reproducible directory structure.
    """

    def build(
        self,
        intent_spec: ScientificIntentSpec,
        md_ir: MDTypedIR | None,
        diagnostics: list[Diagnostic],
        lammps_script: str,
        preflight_report: PreflightReport,
        repair_history: list | None = None,
    ) -> MDSynthOutput:
        """
        Build the complete evidence package.

        Args:
            intent_spec: The scientific intent specification
            md_ir: The final MD Typed IR (None for direct LLM generation path)
            diagnostics: All validation diagnostics (empty for direct path)
            lammps_script: The generated LAMMPS input script
            preflight_report: Sandbox preflight results
            repair_history: Repair actions (list[RepairAction] for template path,
                           list[dict] for direct generation path)

        Returns:
            MDSynthOutput with all package contents
        """
        output = MDSynthOutput()
        is_direct_gen = md_ir is None

        # Serialize core artifacts
        output.lammps_script = lammps_script
        output.md_ir_yaml = to_yaml(md_ir) if md_ir is not None else ""
        output.intent_spec_yaml = to_yaml(intent_spec)

        # Build validation report
        if not is_direct_gen:
            errors = [d for d in diagnostics if d.severity.value == "error"]
            warnings = [d for d in diagnostics if d.severity.value == "warning"]
            infos = [d for d in diagnostics if d.severity.value == "info"]

            validation_report = {
                "static_checks": {
                    "passed": len(infos),
                    "warnings": len(warnings),
                    "failed": len(errors),
                },
                "errors": [
                    {"rule_id": d.rule_id, "message": d.message, "path": d.path}
                    for d in errors
                ],
                "warnings": [
                    {"rule_id": d.rule_id, "message": d.message, "path": d.path}
                    for d in warnings
                ],
            }
            output.validation_report_json = to_json(validation_report)
        else:
            output.validation_report_json = to_json({
                "note": "Direct LLM generation path — no IR-level validation performed",
            })

        # Serialize preflight report
        output.preflight_report_json = to_json(preflight_report)

        # Build provenance manifest
        if not is_direct_gen:
            manifest = self._build_manifest(
                intent_spec=intent_spec,
                md_ir=md_ir,
                diagnostics=diagnostics,
                preflight_report=preflight_report,
                repair_history=repair_history or [],
            )
        else:
            manifest = self._build_direct_gen_manifest(
                intent_spec=intent_spec,
                preflight_report=preflight_report,
                repair_history=repair_history or [],
            )
        output.manifest = manifest
        output.provenance_lock_yaml = to_yaml(manifest)

        # Build human-readable documents
        output.readme_md = self._build_readme(intent_spec, md_ir, preflight_report)
        output.assumptions_md = self._build_assumptions_md(
            intent_spec, md_ir, repair_history or []
        )
        output.semantic_diff_md = self._build_semantic_diff_md(repair_history or [])
        output.repair_history_md = self._build_repair_history_md(repair_history or [], is_direct_gen)

        # Determine success
        output.success = preflight_report.passed

        return output

    def write_to_directory(self, output: MDSynthOutput, directory: Path) -> None:
        """
        Write the complete output package to a directory.

        Args:
            output: The MDSynthOutput to write
            directory: Target directory (created if needed)
        """
        directory.mkdir(parents=True, exist_ok=True)

        # LAMMPS script (always)
        (directory / "in.main.lammps").write_text(output.lammps_script, encoding="utf-8")

        # Serialized IRs (skip if empty — direct generation path)
        if output.md_ir_yaml:
            (directory / "md_ir.yaml").write_text(output.md_ir_yaml, encoding="utf-8")
        if output.intent_spec_yaml:
            (directory / "intent_spec.yaml").write_text(output.intent_spec_yaml, encoding="utf-8")

        # Reports
        if output.validation_report_json:
            (directory / "validation_report.json").write_text(
                output.validation_report_json, encoding="utf-8"
            )
        (directory / "preflight_report.json").write_text(
            output.preflight_report_json, encoding="utf-8"
        )
        if output.provenance_lock_yaml:
            (directory / "provenance.lock").write_text(
                output.provenance_lock_yaml, encoding="utf-8"
            )

        # Documents
        if output.readme_md:
            (directory / "README.md").write_text(output.readme_md, encoding="utf-8")
        if output.assumptions_md:
            (directory / "assumptions.md").write_text(output.assumptions_md, encoding="utf-8")
        semantic_diff_path = directory / "semantic_diff.md"
        if output.semantic_diff_md:
            semantic_diff_path.write_text(output.semantic_diff_md, encoding="utf-8")
        elif semantic_diff_path.exists():
            semantic_diff_path.unlink()

        # Repair history (direct generation path)
        repair_history_path = directory / "repair_history.md"
        if output.repair_history_md:
            repair_history_path.write_text(output.repair_history_md, encoding="utf-8")
        elif repair_history_path.exists():
            repair_history_path.unlink()

    def _build_manifest(
        self,
        intent_spec: ScientificIntentSpec,
        md_ir: MDTypedIR,
        diagnostics: list[Diagnostic],
        preflight_report: PreflightReport,
        repair_history: list[RepairAction],
    ) -> ProvenanceManifest:
        """Build provenance manifest from all artifacts."""
        # Force field provenance
        ff = md_ir.force_field
        ff_prov = ForceFieldProvenance(
            family=ff.family.value,
            source=ff.provenance or "builtin",
            verification=ff.verification_status,
            potential_files=[pf.name for pf in ff.potential_files],
            elements_covered=ff.elements_covered,
        )

        # System info
        replication = (10, 10, 10)
        if (
            md_ir.system.structure
            and md_ir.system.structure.generator
        ):
            replication = md_ir.system.structure.generator.replication

        return ProvenanceManifest(
            user_request=intent_spec.original_user_text,
            task_type=intent_spec.task_type,
            material=intent_spec.material.name or "",
            crystal_structure=intent_spec.material.crystal_structure or "",
            lattice_constant=(
                md_ir.system.structure.generator.lattice_constant.value
                if md_ir.system.structure
                and md_ir.system.structure.generator
                else 0.0
            ),
            force_field_provenance=ff_prov,
            assumptions=intent_spec.assumptions,
            limitations=intent_spec.risk_flags,
            repair_actions=[
                {"id": a.id, "permission": a.permission.value, "reason": a.reason}
                for a in repair_history
            ],
            num_atoms=md_ir.system.total_atoms or 0,
            replication=replication,
            num_stages=len(md_ir.stages),
            stage_ids=[s.id for s in md_ir.stages],
            validation_passed=not any(
                d.severity.value == "error" for d in diagnostics
            ),
            preflight_passed=preflight_report.passed,
        )

    def _build_direct_gen_manifest(
        self,
        intent_spec: ScientificIntentSpec,
        preflight_report: PreflightReport,
        repair_history: list,
    ) -> ProvenanceManifest:
        """Build provenance manifest for direct generation path (no IR)."""
        from mdsynth.knowledge.materials import get_material as get_mat

        mat_name = intent_spec.material.name or "unknown"
        mat_data = get_mat(mat_name)

        ff_prov = ForceFieldProvenance(
            family="eam",
            source="llm_direct",
            verification="unverified",
            potential_files=(
                [intent_spec.user_potential_path]
                if intent_spec.user_potential_path
                else []
            ),
            elements_covered=[mat_data.symbol] if mat_data else [],
        )

        rounds_info = [
            {
                "round": h.get("round", i + 1),
                "passed": h.get("passed", False),
                "error": h.get("error", "")[:200] if h.get("error") else None,
            }
            for i, h in enumerate(repair_history)
        ]

        return ProvenanceManifest(
            user_request=intent_spec.original_user_text,
            task_type="llm_direct_generation",
            material=mat_name,
            crystal_structure=mat_data.crystal_structure if mat_data else "",
            lattice_constant=mat_data.lattice_constant if mat_data else 0.0,
            force_field_provenance=ff_prov,
            assumptions=intent_spec.assumptions + [
                "Script generated directly by LLM (no template match)",
                f"Repair rounds: {len(repair_history)}",
            ],
            limitations=intent_spec.risk_flags,
            repair_actions=rounds_info,
            num_atoms=0,
            replication=(0, 0, 0),
            num_stages=0,
            stage_ids=[],
            validation_passed=False,
            preflight_passed=preflight_report.passed,
        )

    def _build_readme(
        self,
        intent_spec: ScientificIntentSpec,
        md_ir: MDTypedIR | None,
        preflight_report: PreflightReport,
    ) -> str:
        """Build README.md. Handles both template and direct generation paths."""
        from mdsynth.intent.taxonomy import MVP_TASK_TYPES
        from mdsynth.knowledge.materials import get_material as get_mat

        mat_name = intent_spec.material.name or "unknown"
        mat_data = get_mat(mat_name)

        if md_ir is None:
            # Direct generation path — simplified README
            return f"""# MDSynth Simulation Output (Direct LLM Generation)

## Task
{intent_spec.original_user_text or intent_spec.task_description}

## Material
- **Name**: {mat_name}
- **Symbol**: {mat_data.symbol if mat_data else '?'}
- **Structure**: {mat_data.crystal_structure if mat_data else '?'}
- **Lattice Constant**: {mat_data.lattice_constant if mat_data else 'N/A'} Å

## Generation Method
This script was generated directly by LLM (no template match).
Preflight run-0 check: **{preflight_report.summary()}**

## Assumptions
{chr(10).join(f'- {a}' for a in intent_spec.assumptions) if intent_spec.assumptions else '- None recorded'}

## Limitations
{chr(10).join(f'- {r}' for r in intent_spec.risk_flags) if intent_spec.risk_flags else '- None flagged'}
"""

        # Template path — full README
        task_info = MVP_TASK_TYPES.get(intent_spec.task_type, {})
        task_display = task_info.get("display_name", intent_spec.task_type)

        protocol_lines = []
        for stage in md_ir.stages:
            line = f"- **{stage.id}** ({stage.type.value})"
            if stage.ensemble:
                line += f": {stage.ensemble.type.value}"
                if stage.ensemble.temperature:
                    line += f" at {stage.ensemble.temperature.value} K"
                if stage.duration:
                    line += f" for {stage.duration.value} {stage.duration.unit}"
            protocol_lines.append(line)

        timestep = md_ir.timestep.value if md_ir.timestep else 0.001

        return build_readme(
            task_display_name=task_display,
            generated_at="",
            task_type=intent_spec.task_type,
            material_name=mat_name,
            material_symbol=mat_data.symbol if mat_data else "?",
            crystal_structure=mat_data.crystal_structure if mat_data else "?",
            lattice_constant=mat_data.lattice_constant if mat_data else 0.0,
            user_request=intent_spec.original_user_text or intent_spec.task_description,
            num_atoms=md_ir.system.total_atoms or 0,
            replication_x=10,
            replication_y=10,
            replication_z=10,
            force_field_family=md_ir.force_field.family.value,
            potential_file=(
                md_ir.force_field.potential_files[0].name
                if md_ir.force_field.potential_files
                else "N/A"
            ),
            timestep=timestep,
            timestep_fs=timestep * 1000,
            num_stages=len(md_ir.stages),
            protocol_summary="\n".join(protocol_lines) if protocol_lines else "N/A",
            assumptions="\n".join(
                f"- {a}" for a in intent_spec.assumptions
            ) if intent_spec.assumptions else "- None",
            limitations="\n".join(
                f"- {r}" for r in intent_spec.risk_flags
            ) if intent_spec.risk_flags else "- None",
            static_validation="PASSED" if not any(
                d.severity.value == "error" for d in []
            ) else "FAILED",
            preflight_status=preflight_report.summary(),
        )

    def _build_assumptions_md(
        self,
        intent_spec: ScientificIntentSpec,
        md_ir: MDTypedIR | None,
        repair_history: list,
    ) -> str:
        """Build assumptions.md. Handles both template and direct generation paths."""
        if md_ir is None:
            force_field_notes = "- LLM-chosen potential (direct generation path)"
        else:
            force_field_notes = "\n".join(
                f"- {n}" for n in md_ir.force_field.notes
            ) if md_ir.force_field.notes else "- Default EAM potential for this material"

        return build_assumptions(
            assumptions_list="\n".join(
                f"- {a}" for a in intent_spec.assumptions
            ) if intent_spec.assumptions else "- None recorded",
            risks_list="\n".join(
                f"- ⚠ {r}" for r in intent_spec.risk_flags
            ) if intent_spec.risk_flags else "- None flagged",
            defaults_list="- System defaults applied for unspecified parameters (see provenance.lock)",
            force_field_notes=force_field_notes,
            repair_history=self._format_repair_history(repair_history),
        )

    def _build_semantic_diff_md(
        self,
        repair_history: list,
    ) -> str:
        """Build semantic_diff.md. Handles both RepairAction and direct-gen dict formats."""
        if not repair_history:
            return ""

        # Check if this is direct-gen repair history (list of dicts)
        if isinstance(repair_history[0], dict):
            entries = []
            for h in repair_history:
                status = "✅ PASSED" if h.get("passed") else "❌ FAILED"
                entries.append(
                    f"- **Round {h.get('round', '?')}** {status}"
                )
                if not h.get("passed") and h.get("error"):
                    err = h["error"][:200]
                    entries.append(f"  ```\n  {err}\n  ```")
            return build_semantic_diff(
                repair_entries="\n".join(entries),
                total_repairs=len(repair_history),
                count_a=sum(1 for h in repair_history if h.get("passed")),
                count_b=sum(1 for h in repair_history if not h.get("passed")),
                count_c=0,
                count_d=0,
                count_e=0,
            )

        # Template path: list of RepairAction
        count_a = sum(1 for a in repair_history if a.permission.value == "A")
        count_b = sum(1 for a in repair_history if a.permission.value == "B")
        count_c = sum(1 for a in repair_history if a.permission.value == "C")
        count_d = sum(1 for a in repair_history if a.permission.value == "D")
        count_e = sum(1 for a in repair_history if a.permission.value == "E")

        return build_semantic_diff(
            repair_entries=self._format_repair_history(repair_history),
            total_repairs=len(repair_history),
            count_a=count_a,
            count_b=count_b,
            count_c=count_c,
            count_d=count_d,
            count_e=count_e,
        )

    def _format_repair_history(self, repair_history: list) -> str:
        """Format repair history for markdown output."""
        if not repair_history:
            return "No repairs were needed."

        lines = []
        for item in repair_history:
            if isinstance(item, dict):
                status = "✅" if item.get("passed") else "❌"
                lines.append(
                    f"- **Round {item.get('round', '?')}** {status}"
                )
                if item.get("error"):
                    lines.append(f"  Error: {item['error'][:150]}")
            else:
                lines.append(
                    f"- **{item.id}** [Permission {item.permission.value}] "
                    f"— {item.reason}"
                )
        return "\n".join(lines) if lines else "No repairs were needed."

    def _build_repair_history_md(
        self,
        repair_history: list,
        is_direct_gen: bool = False,
    ) -> str:
        """Build repair_history.md — full LLM modification trajectory for direct-gen path."""
        if not is_direct_gen or not repair_history:
            return ""

        # Check if this is direct-gen dict format
        if not isinstance(repair_history[0], dict):
            return ""

        lines = [
            "# LLM Direct Generation — Repair Trajectory",
            "",
            f"Total rounds: **{len(repair_history)}**",
            "",
        ]

        all_passed = any(h.get("passed") for h in repair_history)
        if all_passed:
            passed_round = next(h["round"] for h in repair_history if h.get("passed"))
            lines.append(f"✅ **PASSED** at round {passed_round}")
        else:
            lines.append("❌ **FAILED** — all rounds exhausted")
        lines.append("")
        lines.append("---")
        lines.append("")

        for h in repair_history:
            round_num = h.get("round", "?")
            passed = h.get("passed", False)
            status_icon = "✅" if passed else "❌"
            script = h.get("script", "")
            error = h.get("error", "")

            lines.append(f"## Round {round_num} {status_icon}")
            lines.append("")

            generation_refs = h.get("generation_retrieval", [])
            if generation_refs:
                lines.append("### Retrieved Script References")
                lines.append("")
                for ref in generation_refs:
                    lines.append(
                        f"- {ref.get('title') or ref.get('id') or 'Untitled'} "
                        f"(score: {ref.get('score', 'n/a')})"
                    )
                lines.append("")

            lines.append("### Generated Script")
            lines.append("")
            lines.append("```lammps")
            lines.append(script if script else "(empty)")
            lines.append("```")
            lines.append("")

            if not passed and error:
                lines.append("### LAMMPS Error")
                lines.append("")
                lines.append("```")
                lines.append(error[:3000])
                lines.append("```")
                lines.append("")

            repair_refs = h.get("repair_retrieval", [])
            if repair_refs:
                lines.append("### Retrieved Command References")
                lines.append("")
                for ref in repair_refs:
                    label = (
                        ref.get("command_name")
                        or ref.get("title")
                        or ref.get("id")
                        or "Untitled"
                    )
                    source = ref.get("source_url") or ""
                    lines.append(f"- {label}" + (f" — {source}" if source else ""))
                lines.append("")

            lines.append("---")
            lines.append("")

        return "\n".join(lines)
