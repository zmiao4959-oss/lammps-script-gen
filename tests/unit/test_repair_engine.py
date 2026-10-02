"""
Tests for ConstrainedRepairEngine.
"""

import pytest

from mdsynth.repair.engine import ConstrainedRepairEngine, RepairAction
from mdsynth.repair.permissions import RepairPermission
from mdsynth.validator.diagnostic import Diagnostic, Severity


class TestRepairEngine:
    """Test the constrained repair engine."""

    @pytest.fixture
    def engine(self):
        return ConstrainedRepairEngine(max_auto_rounds=3)

    def test_generate_actions_for_a_class(self, engine):
        """A-class errors should get auto-repair actions."""
        diag = Diagnostic(
            rule_id="TEST_A",
            severity=Severity.ERROR,
            message="Test A-class error",
            path="test.path",
            repair_permission=RepairPermission.A,
            repair_action={"action_type": "set_value", "new_value": "fixed"},
        )
        actions = engine.generate_actions(None, [diag])
        assert len(actions) == 1
        assert actions[0].permission == RepairPermission.A

    def test_generate_actions_for_b_class(self, engine):
        """B-class errors should get auto-repair actions."""
        diag = Diagnostic(
            rule_id="TEST_B",
            severity=Severity.ERROR,
            message="Test B-class error",
            path="test.path",
            repair_permission=RepairPermission.B,
            repair_action={"action_type": "set_value", "new_value": "fixed"},
        )
        actions = engine.generate_actions(None, [diag])
        assert len(actions) == 1
        assert actions[0].permission == RepairPermission.B

    def test_generate_actions_for_c_class(self, engine):
        """C-class errors should get candidate actions."""
        diag = Diagnostic(
            rule_id="TEST_C",
            severity=Severity.ERROR,
            message="Test C-class error",
            path="test.path",
            repair_permission=RepairPermission.C,
            repair_action={"action_type": "multiply", "new_value": 0.5},
        )
        actions = engine.generate_actions(None, [diag])
        assert len(actions) == 1
        assert actions[0].confidence == "medium"

    def test_generate_actions_for_d_class(self, engine):
        """D-class errors should be blocked."""
        diag = Diagnostic(
            rule_id="TEST_D",
            severity=Severity.ERROR,
            message="Test D-class error",
            path="test.path",
            repair_permission=RepairPermission.D,
        )
        actions = engine.generate_actions(None, [diag])
        assert len(actions) == 1
        assert actions[0].action_type == "blocked"

    def test_generate_actions_for_e_class(self, engine):
        """E-class errors should be blocked."""
        diag = Diagnostic(
            rule_id="TEST_E",
            severity=Severity.ERROR,
            message="Test E-class error",
            path="test.path",
            repair_permission=RepairPermission.E,
        )
        actions = engine.generate_actions(None, [diag])
        assert len(actions) == 1
        assert actions[0].action_type == "blocked"

    def test_history_recorded(self, engine):
        """Repair history should be recorded."""
        diag = Diagnostic(
            rule_id="TEST",
            severity=Severity.ERROR,
            message="Test error",
            path="timestep.value",
            repair_permission=RepairPermission.A,
            repair_action={"action_type": "set_value", "new_value": 0.0005},
        )
        actions = engine.generate_actions(None, [diag])

        # We need a real IR to apply
        from mdsynth.ir.md_ir import MDTypedIR
        from mdsynth.utils.quantities import Quantity, Dimension

        ir = MDTypedIR(timestep=Quantity(0.01, "ps", Dimension.TIME))
        engine.apply_actions(ir, actions)

        history = engine.get_history()
        assert len(history) == 1
