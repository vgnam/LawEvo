import numpy as np
import pytest

from lawevo.pid import ADAPTERS, CONTROLLED_VARIANT_ADAPTERS, PANDA_GYM_ADAPTERS
from lawevo.pid.expression import SymbolicExpression as Law
from lawevo.pid.gym_benchmark import BenchmarkAdapter, GymMetrics, tune_gym_cem
from lawevo.pid.signal_schema import SignalContract, SignalSpec, validate_expression


@pytest.mark.parametrize("text", [
    "K1*(x+y)", "K1*(x+y)*(x-y)", "K1*x + K1*y + K2*z",
    "K1*min(x+y, K2*z)", "K1*(K2*x+K3*y)",
    "K1*(x + 0.12345678901234567)",
])
def test_ast_text_tree_and_rename_preserve_actions(text):
    law = Law("original", text)
    payload = law.to_dict()
    values = {"x": np.array([0.5, -1.2]), "y": np.array([2.0, 0.1]),
              "z": np.array([-1.0, 3.0])}
    gains = np.arange(1, law.parameter_count + 1) * 1.3
    candidates = [Law("text", law.to_expression_string()), Law("tree", payload["tree"]),
                  Law.from_dict(payload), law.renamed("new name")]
    payload["expression"] = "K1*x"  # v2 AST is authoritative
    candidates.append(Law.from_dict(payload))
    for candidate in candidates:
        assert candidate.parameter_count == law.parameter_count
        assert candidate.key() == law.key()
        np.testing.assert_allclose(candidate.evaluate(values, gains), law.evaluate(values, gains))


def test_key_preserves_sharing_but_ignores_gain_names_and_operand_order():
    a = Law("a", "K1*x + K1*y + K2*z")
    b = Law("b", "K1*x + K2*y + K2*z")
    c = Law("c", "K8*z + K9*y + K9*x")
    assert a.key() != b.key()
    assert a.key() == c.key()
    a = Law("a", "K1*x*K2*y + K1*y*K2*x")
    b = Law("b", "K8*x*K9*y + K9*x*K8*y")
    assert a.key() == b.key()
    assert Law("a", "K1*x + 0.1234567891").key() != Law("b", "K1*x + 0.1234567892").key()


def test_legacy_string_sharing_and_new_parameter_limit():
    law = Law.from_dict({"name": "old", "expression": "K1*x+K1*y"})
    assert law.parameter_count == 1
    with pytest.raises(ValueError, match="12 gain slots"):
        Law("too many", "+".join(f"K{i}*x" for i in range(13)))


SUITE = {**CONTROLLED_VARIANT_ADAPTERS, **{key: ADAPTERS[key] for key in
         ("inverted_pendulum", "reacher")}, **{key: PANDA_GYM_ADAPTERS[key] for key in
         ("panda_reach", "panda_push", "panda_slide")}}


@pytest.mark.parametrize("key", tuple(SUITE))
def test_all_ten_tasks_have_complete_contracts_and_dimensionally_valid_baselines(key):
    adapter = SUITE[key]
    contract = adapter.signal_contract
    assert contract.complete
    assert set(contract.signals) == set(adapter.allowed_terms)
    for spec in contract.signals.values():
        assert all((spec.unit, spec.formula, spec.sign, spec.zero_when,
                    spec.value_range, spec.memory, spec.action_mapping))
    for law in adapter.classical:
        assert adapter.validate_structure(law)["checked"]


def test_gain_unit_inference_rejects_bad_addition_tying_and_nonlinearity():
    contract = PANDA_GYM_ADAPTERS["panda_reach"].signal_contract
    for text in ("K1*(goal_error+eef_damping)",
                 "K1*goal_error+K1*eef_damping", "K1*tanh(goal_error)"):
        with pytest.raises(ValueError, match="inconsistent units"):
            validate_expression(Law("invalid", text), contract)
    units = validate_expression(Law("PD", "K1*goal_error+K2*eef_damping"), contract)
    np.testing.assert_allclose(units["gain_unit_exponents_m_s"], [[-1, 0], [-1, 1]], atol=1e-8)
    assert validate_expression(Law("sat", "K1*tanh(K2*goal_error)"), contract)["checked"]
    assert validate_expression(Law("gate", "max(0,K1*goal_error)"), contract)["checked"]


def test_scalar_vector_broadcast_and_shape_mismatch():
    def spec(shape):
        return SignalSpec("normalized", "1", (0, 0), shape, "test", "positive",
                          "input zero", "unbounded")
    contract = SignalContract({"s": spec(()), "v": spec((3,)), "bad": spec((2,))},
                              (3,), "xyz")
    assert validate_expression(Law("broadcast", "K1*s*v"), contract)["checked"]
    with pytest.raises(ValueError, match="incompatible signal shapes"):
        validate_expression(Law("mismatch", "K1*bad*v"), contract)
    with pytest.raises(ValueError, match="expected finite shape"):
        contract.validate_values({"s": 1.0, "v": [1, 2], "bad": [1, 2]})


def metric(sr, sg, score):
    return GymMetrics(score, 0, sr, 0, 0, 1, sg)


def test_success_cannot_be_traded_for_arbitrary_secondary_utility():
    assert metric(0.8, 1.0, -1e100).selection_key > metric(0.79, 0, 1e100).selection_key
    assert metric(0.8, 0.1, -1e100).selection_key > metric(0.8, 0.2, 1e100).selection_key
    assert metric(0.8, 0.1, 0.2).selection_key > metric(0.8, 0.1, 0.1).selection_key
    assert metric(0.5, None, 0).selection_key > metric(1, 0, float("nan")).selection_key


def test_batched_cem_preserves_success_over_secondary_score():
    class Adapter(BenchmarkAdapter):
        env_id = "fake"
        allowed_terms = ("x",)

        def evaluate_gain_batch(self, structure, gains, seeds):
            if len(gains) == 1:
                return [metric(0.5, 0.1, 0)]
            return [metric(0.75, 0.1, -1e9), metric(0.25, 0, 1e9)]

    _, metrics = tune_gym_cem(Adapter(), Law("test", "K1*x"), [1],
                              iterations=1, population_size=2)
    assert metrics.success_rate == 0.75
    assert metrics.score == -1e9


def test_cpu_cem_uses_same_success_first_order(monkeypatch):
    import lawevo.pid.gym_benchmark as benchmark

    class Env:
        def close(self):
            pass

    class Adapter(BenchmarkAdapter):
        env_id = "fake"
        allowed_terms = ("x",)

        def make_env(self):
            return Env()

    results = iter([metric(0.5, 0.1, 0), metric(0.75, 0.1, -1e9), metric(0.25, 0, 1e9)])
    monkeypatch.setattr(benchmark, "evaluate_gym_structure", lambda *a, **k: (next(results), []))
    _, metrics = tune_gym_cem(Adapter(), Law("test", "K1*x"), [1],
                              iterations=1, population_size=2)
    assert metrics.success_rate == 0.75


def test_prompt_and_rejections_use_contract_and_objective():
    from experiments.gymnasium_classical_benchmarks import extract_structures, prompt

    adapter = PANDA_GYM_ADAPTERS["panda_reach"]
    errors = []
    laws = extract_structures('[{"name":"bad","expression":"K1*tanh(goal_error)"}]',
                              adapter.allowed_terms, adapter.signal_contract, errors)
    assert not laws and "units" in errors[0]["reason"]
    text = prompt("panda_reach", adapter.allowed_terms, [], [], 2, 1, errors)
    assert "-eef_linear_velocity" in text
    assert "Exact selection is lexicographic" in text
    assert "inconsistent units" in text
    assert "Exact scalar fitness is the environment return" not in text


def test_canonicalization_handles_twelve_interchangeable_shared_gains():
    forward = "*".join(f"K{i}" for i in range(12))
    reverse = "*".join(f"K{i}" for i in reversed(range(12)))
    a = Law("a", f"{forward}*x + {forward}*y")
    b = Law("b", f"{reverse}*y + {reverse}*x")
    assert a.key() == b.key()
