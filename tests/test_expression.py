import numpy as np
import pytest

from lawevo.pid.expression import MAX_NODES, SymbolicExpression


def test_legacy_terms_build_linear_expression() -> None:
    law = SymbolicExpression("P", ("angle", "integral_angle"))
    assert law.signals == ("angle", "integral_angle")
    assert law.parameter_count == 2
    assert law.to_expression_string() == "K1*angle + K2*integral_angle"
    assert law.complexity == 3


def test_linear_classmethod_matches_legacy_form() -> None:
    law = SymbolicExpression.linear("PD", ["angle", "angular_velocity"])
    other = SymbolicExpression("PD", ("angle", "angular_velocity"))
    assert law.key() == other.key()


def test_expression_string_evaluates_expected_values() -> None:
    law = SymbolicExpression("swing", "K1*tanh(K2*angle) + K3*angle*angular_velocity")
    values = {"angle": np.array([0.5]), "angular_velocity": np.array([2.0])}
    gains = np.array([2.0, 1.0, -0.5])
    expected = 2.0 * np.tanh(1.0 * 0.5) + (-0.5) * 0.5 * 2.0
    result = law.evaluate(values, gains)
    assert result.shape == (1,)
    assert np.allclose(result, [expected])


def test_expression_vectorized_over_action_dimension() -> None:
    law = SymbolicExpression("pd", "K1*angle + K2*min(angle, angular_velocity)")
    values = {
        "angle": np.array([0.1, -0.2, 0.3]),
        "angular_velocity": np.array([0.5, 0.1, -1.0]),
    }
    gains = np.array([1.0, 2.0])
    result = law.evaluate(values, gains)
    expected = np.array(
        [
            1.0 * 0.1 + 2.0 * min(0.1, 0.5),
            1.0 * -0.2 + 2.0 * min(-0.2, 0.1),
            1.0 * 0.3 + 2.0 * min(0.3, -1.0),
        ]
    )
    assert np.allclose(result, expected)


def test_repeated_gain_name_shares_one_slot() -> None:
    law = SymbolicExpression("tied", "K1*angle + K1*angular_velocity")
    assert law.parameter_count == 1
    result = law.evaluate(
        {"angle": np.array([1.0]), "angular_velocity": np.array([2.0])}, [3.0]
    )
    assert np.allclose(result, [9.0])
    # Round-trip must preserve the tied slot.
    restored = SymbolicExpression.from_dict(law.to_dict())
    assert restored.parameter_count == 1
    assert restored.key() == law.key()


def test_key_is_order_insensitive_for_sums_and_products() -> None:
    first = SymbolicExpression("a", "K1*angle + K2*angular_velocity")
    second = SymbolicExpression("b", "K2*angular_velocity + K1*angle")
    assert first.key() == second.key()
    product_a = SymbolicExpression("c", "K1*angle*K2*angular_velocity")
    product_b = SymbolicExpression("d", "K2*angular_velocity*K1*angle")
    assert product_a.key() == product_b.key()


def test_key_distinguishes_tied_from_distinct_gains() -> None:
    tied = SymbolicExpression("a", "K1*angle + K1*angular_velocity")
    free = SymbolicExpression("b", "K1*angle + K2*angular_velocity")
    assert tied.key() != free.key()
    assert tied.parameter_count == 1
    assert free.parameter_count == 2


def test_to_dict_round_trip_preserves_structure() -> None:
    law = SymbolicExpression(
        "rich", "K1*tanh(K2*sqrt(angle)) + min(K3*angle, K4*angular_velocity) - 0.5"
    )
    restored = SymbolicExpression.from_dict(law.to_dict())
    assert restored.name == law.name
    assert restored.key() == law.key()
    assert restored.parameter_count == law.parameter_count
    assert restored.complexity == law.complexity
    values = {"angle": np.array([0.3, -0.7]), "angular_velocity": np.array([1.0, 0.2])}
    gains = np.arange(1.0, 5.0)
    assert np.allclose(
        law.evaluate(values, gains),
        restored.evaluate(values, gains),
    )


def test_formula_substitutes_gain_values() -> None:
    law = SymbolicExpression("pd", "K1*angle + K2*angular_velocity")
    text = law.formula(np.array([1.5, -2.0]))
    assert "(1.5)*angle" in text
    assert "(-2)*angular_velocity" in text


def test_evaluator_guards_divide_by_zero_and_overflow() -> None:
    law = SymbolicExpression("safe", "K1*exp(K2*angle) + K3*square(K4*angle)")
    values = {"angle": np.array([200.0, -200.0])}
    result = law.evaluate(values, [1.0, 1.0, 1.0, 1.0])
    assert np.all(np.isfinite(result))


def test_validate_rejects_unknown_signals() -> None:
    law = SymbolicExpression("bad", "K1*angle + K2*mystery")
    with pytest.raises(ValueError):
        law.validate(("angle", "angular_velocity"))
    law.validate(("angle", "mystery"))


@pytest.mark.parametrize(
    "expression",
    [
        "K1",
        "angle",
        "K1*tanh(K1*tanh(K1*tanh(K1*tanh(K1*tanh(K1*angle)))))",
        " + ".join(f"K{i}*sig{i}" for i in range(1, MAX_NODES + 2)),
    ],
)
def test_invalid_expressions_are_rejected(expression: str) -> None:
    names = [f"sig{i}" for i in range(1, MAX_NODES + 3)] + ["angle"]
    with pytest.raises(ValueError):
        SymbolicExpression("bad", expression).validate(names)


def test_from_dict_accepts_legacy_terms_payload() -> None:
    law = SymbolicExpression.from_dict(
        {"name": "old", "terms": ["jt_error", "task_damping"]}
    )
    assert law.to_expression_string() == "K1*jt_error + K2*task_damping"


def test_run_episode_uses_expression_law() -> None:
    from lawevo.pid.gym_benchmark import ADAPTERS, run_episode

    adapter = ADAPTERS["pendulum"]
    law = SymbolicExpression("swing", "K1*tanh(K2*angle) + K3*angular_velocity")
    episode = run_episode(adapter, law, np.array([2.0, 1.5, -0.4]), seed=7)
    assert np.isfinite(episode.episode_return)
