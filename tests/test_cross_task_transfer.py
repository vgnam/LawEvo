import copy

from experiments.cross_task_transfer import compatibility


def contract():
    return {
        "complete": True,
        "action_shape": [2],
        "action": "motor commands",
        "signals": {"error": {"unit": "m", "sign": "target-current"}},
    }


def test_transfer_rejects_same_name_with_changed_semantics():
    source = contract()
    target = copy.deepcopy(source)
    assert compatibility(source, target, ["error"]) is None
    target["signals"]["error"]["sign"] = "current-target"
    assert "differs" in compatibility(source, target, ["error"])


def test_transfer_rejects_missing_signals_and_unknown_contracts():
    source = contract()
    assert "unavailable" in compatibility(source, source, ["waypoint_push"])
    target = copy.deepcopy(source)
    target["complete"] = False
    assert "incomplete" in compatibility(source, target, ["error"])


def test_transfer_rejects_changed_action_dimension():
    source = contract()
    target = copy.deepcopy(source)
    target["action_shape"] = [3]
    assert "action_shape" in compatibility(source, target, ["error"])
