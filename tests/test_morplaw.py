import json

import numpy as np
import pytest

pytest.importorskip("gymnasium", reason="MorpLaw tests need the benchmarks extras")
pytest.importorskip("mujoco")

import gymnasium as gym
import mujoco

from lawevo.evolve.belief import BeliefSpace, Experience
from lawevo.morplaw import (
    AntTemplate,
    AntTopologyTemplate,
    HalfCheetahTemplate,
    HopperTemplate,
    MorpLawConfig,
    MorpLawRunner,
    ReacherTemplate,
    SwimmerTemplate,
    SwimmerTopologyTemplate,
    Walker2dTemplate,
    evaluate_pair,
    make_morph_env,
    morph_cost,
    tune_pair_cem,
)
from lawevo.morplaw.morphology import MorphologyError, MorphologySpec
from lawevo.pid.gym_benchmark import ADAPTERS, LOCOMOTION_ADAPTERS, GymStructure

REACHER = ADAPTERS["reacher"]


def _reacher_stub_law(prefix: str = "stub"):
    def law_generator(incumbent, belief, count, generation):
        terms_pool = list(REACHER.allowed_terms)
        return [
            GymStructure(
                f"{prefix}_{generation}_{index}",
                tuple(terms_pool[(generation + index) % len(terms_pool) :][:2]),
            )
            for index in range(count)
        ]

    return law_generator


def _reacher_stub_morph(template: ReacherTemplate):
    def morph_generator(incumbent, belief, count, generation):
        specs = []
        for index in range(count):
            values = template.defaults()
            values["gear"] = 200.0 * (1.0 + 0.1 * (generation + index))
            specs.append(MorphologySpec.of(values))
        return specs

    return morph_generator


def _small_config(**overrides) -> MorpLawConfig:
    base = {
        "generations": 2,
        "proposals_per_side": 2,
        "joint_top_k": 1,
        "cem_iterations": 1,
        "cem_population": 4,
    }
    base.update(overrides)
    return MorpLawConfig(**base)


def _run_reacher(config, template, law_prefix="stub", archive=None):
    law_gen = _reacher_stub_law(law_prefix)
    morph_gen = _reacher_stub_morph(template)
    runner = MorpLawRunner(
        REACHER, template, [0, 1], law_gen, morph_gen, config, archive=archive
    )
    initial = [(template.default_spec(), structure) for structure in REACHER.classical[:2]]
    return runner, runner.run(initial)


# --- template / morphology -------------------------------------------------


def test_templates_compile_and_dims_match_default_env() -> None:
    for template, env_id in [
        (Walker2dTemplate(), "Walker2d-v5"),
        (ReacherTemplate(), "Reacher-v5"),
    ]:
        model = template.compile(template.default_spec())
        default_env = gym.make(env_id, max_episode_steps=50)
        morph_env = gym.make(
            env_id, xml_file=str(template.xml_path(template.default_spec())), max_episode_steps=50
        )
        try:
            assert (model.nq, model.nv, model.nu) == (
                default_env.unwrapped.model.nq,
                default_env.unwrapped.model.nv,
                default_env.unwrapped.model.nu,
            )
            assert morph_env.observation_space.shape == default_env.observation_space.shape
            assert morph_env.action_space.shape == default_env.action_space.shape
        finally:
            morph_env.close()
            default_env.close()


def test_walker2d_geometry_coupling() -> None:
    template = Walker2dTemplate()
    values = template.defaults()
    values["thigh_len"] = 0.3
    long_thigh = template.compile(MorphologySpec.of(values))
    values = template.defaults()
    values["leg_len"] = 0.3
    long_leg = template.compile(MorphologySpec.of(values))
    values = template.defaults()
    values["foot_len"] = 0.14
    long_foot = template.compile(MorphologySpec.of(values))
    values = template.defaults()
    values["gear"] = 150.0
    strong_gear = template.compile(MorphologySpec.of(values))
    default_model = template.compile(template.default_spec())

    leg_id = default_model.body("leg").id
    foot_id = default_model.body("foot").id
    # thigh length moves the leg body (coupled through leg_z)
    assert long_thigh.body_pos[leg_id][2] < default_model.body_pos[leg_id][2]
    # leg length moves the foot body (coupled through foot_z)
    assert long_leg.body_pos[foot_id][2] < default_model.body_pos[foot_id][2]
    # foot length shifts the foot geom center ahead of the joint
    assert long_foot.geom("foot_geom").pos[0] > default_model.geom("foot_geom").pos[0]
    assert np.all(strong_gear.actuator_gear[:, 0] == 150.0)
    assert default_model.body("thigh").id >= 0
    assert mujoco.mj_name2id(default_model, mujoco.mjtObj.mjOBJ_BODY, "thigh_left") >= 0


def test_morphology_spec_validation() -> None:
    template = ReacherTemplate()
    out_of_bounds = MorphologySpec.of({**template.defaults(), "gear": 9999.0})
    assert template.validate(out_of_bounds)
    with pytest.raises(MorphologyError):
        template.check(out_of_bounds)
    with pytest.raises(MorphologyError):
        MorphologySpec.of({**template.defaults(), "l0": float("nan")})
    unknown = MorphologySpec.of({**template.defaults(), "mystery": 1.0})
    assert template.validate(unknown)
    missing = MorphologySpec.of({name: 1.0 for name in list(template.defaults())[:-1]})
    assert template.validate(missing)
    spec = MorphologySpec.of(template.defaults())
    assert template.validate(spec) == []
    assert "gear" in spec.describe()


def test_equal_cem_budget_across_pairs() -> None:
    template = ReacherTemplate()
    structure = REACHER.classical[0]
    _, _, budget_default = tune_pair_cem(
        REACHER, template, template.default_spec(), structure, [0, 1],
        iterations=1, population_size=4,
    )
    values = template.defaults()
    values["gear"] = 250.0
    _, _, budget_changed = tune_pair_cem(
        REACHER, template, MorphologySpec.of(values), structure, [0, 1],
        iterations=1, population_size=4,
    )
    assert budget_default == budget_changed == 2 * (1 + 1 * 4)


# --- runner ------------------------------------------------------------------


def test_runner_elitism_determinism_and_archive_dedup() -> None:
    template = ReacherTemplate()
    config = _small_config(cross_direction="both")
    runner_a, (best_a, reports_a) = _run_reacher(config, template)
    scores_a = [report.best_score for report in reports_a]
    assert scores_a == sorted(scores_a)  # elitism: never worsens
    archive_size = len(runner_a.archive)

    runner_b, (best_b, _) = _run_reacher(config, template)
    assert best_a.key() == best_b.key()  # deterministic
    assert len(runner_b.archive) == archive_size

    # Proposing the same pairs again must hit the archive without new evaluations.
    episodes_gen_1 = runner_a.episodes_spent
    assert episodes_gen_1 == runner_b.episodes_spent


def test_direction_gating_and_channel_contexts() -> None:
    template = ReacherTemplate()
    for direction, morph_nonempty, law_nonempty in [
        ("none", False, False),
        ("m_to_l", True, False),
        ("l_to_m", False, True),
        ("both", True, True),
    ]:
        runner, _ = _run_reacher(_small_config(cross_direction=direction), template)
        assert bool(runner.belief.morph_to_law) == morph_nonempty, direction
        assert bool(runner.belief.law_to_morph) == law_nonempty, direction
        for item in runner.belief.morph_to_law:
            assert item.context is not None and set(item.context) == {"morphology"}
            assert isinstance(json.loads(item.context["morphology"]), dict)
            assert not item.hypothesis
        for item in runner.belief.law_to_morph:
            assert item.context is not None and set(item.context) == {"structure"}
            assert isinstance(json.loads(item.context["structure"]), list)
            assert item.hypothesis


def test_archive_dedup_skips_reevaluation() -> None:
    template = ReacherTemplate()
    structure = REACHER.classical[0]

    def constant_law(incumbent, belief, count, generation):
        return [GymStructure("constant", structure.terms) for _ in range(count)]

    def constant_morph(incumbent, belief, count, generation):
        return [template.default_spec() for _ in range(count)]

    config = _small_config(cross_direction="none")
    runner = MorpLawRunner(
        REACHER, template, [0, 1], constant_law, constant_morph, config
    )
    runner.run([(template.default_spec(), structure)])
    size_before = len(runner.archive)
    # Generation 2 proposes the exact pairs generation 1 already evaluated.
    runner.run([(template.default_spec(), structure)])
    assert len(runner.archive) == size_before
    # Fresh runner reusing the archive spends no new episodes.
    runner2 = MorpLawRunner(
        REACHER, template, [0, 1], constant_law, constant_morph, config, archive=runner.archive
    )
    runner2.run([(template.default_spec(), structure)])
    assert runner2.episodes_spent == 0


def test_frozen_flags_produce_single_side_runs() -> None:
    template = ReacherTemplate()
    runner, (best, _) = _run_reacher(_small_config(morphology_frozen=True), template)
    assert all(record.spec.key() == template.default_spec().key() for record in runner.archive.values())
    assert not runner.belief.law_to_morph
    assert runner.calls == {"law": 2, "morph": 0}
    assert best.spec.key() == template.default_spec().key()

    runner, _ = _run_reacher(_small_config(law_frozen=True), template)
    assert not runner.belief.morph_to_law
    assert runner.calls == {"law": 0, "morph": 2}


def test_llm_call_parity_across_direction_variants() -> None:
    template = ReacherTemplate()
    for direction in ("both", "m_to_l", "l_to_m", "none"):
        runner, _ = _run_reacher(_small_config(cross_direction=direction), template)
        assert runner.calls == {"law": 2, "morph": 2}, direction


# --- belief -------------------------------------------------------------------


def test_belief_structured_channels_and_context_filter() -> None:
    belief = BeliefSpace()
    belief.update(
        [
            Experience("morph_to_law", "fact-a", 1.0, context={"morphology": "A"}),
            Experience("morph_to_law", "fact-b", 2.0, context={"morphology": "B"}),
            Experience(
                "law_to_morph", "hypo-x", 0.5, context={"structure": '["x"]'}, hypothesis=True
            ),
        ]
    )
    summary_a = belief.summary(("morph_to_law",), context_match={"morphology": "A"})
    assert "fact-a" in summary_a and "fact-b" not in summary_a
    summary_both = belief.summary(("morph_to_law",))
    assert "fact-a" in summary_both and "fact-b" in summary_both
    law_summary = belief.summary(("law_to_morph",))
    assert "[hypothesis]" in law_summary and "hypo-x" in law_summary


def test_belief_legacy_channels_unchanged() -> None:
    belief = BeliefSpace()
    belief.update([Experience("failure", "boom"), Experience("primitive", "p", 1.0)])
    summary = belief.summary(("failure", "primitive"))
    assert "boom" in summary and "p" in summary
    assert belief.failure == ["boom"]


# --- pair evaluation -----------------------------------------------------------


def test_evaluate_pair_returns_normalized_metrics() -> None:
    template = ReacherTemplate()
    spec = template.default_spec()
    structure = REACHER.classical[0]
    gains, train_metrics, _ = tune_pair_cem(
        REACHER, template, spec, structure, [0, 1], iterations=1, population_size=4
    )
    metrics, episodes = evaluate_pair(
        REACHER, template, spec, structure, gains, [1000, 1001]
    )
    assert len(episodes) == 2
    assert np.isfinite(metrics.score)
    assert metrics.energy_norm == pytest.approx(metrics.energy / metrics.total_mass)
    assert train_metrics.total_mass == pytest.approx(metrics.total_mass)
    assert metrics.to_dict()["morph_cost"] == pytest.approx(0.0)  # default spec = zero cost


# --- Phase 4: parametric suite --------------------------------------------------


PARAMETRIC_SUITE = [
    (HopperTemplate, "Hopper-v5"),
    (HalfCheetahTemplate, "HalfCheetah-v5"),
    (SwimmerTemplate, "Swimmer-v5"),
    (AntTemplate, "Ant-v5"),
]


@pytest.mark.parametrize("template_cls,env_id", PARAMETRIC_SUITE)
def test_parametric_suite_matches_default_env(template_cls, env_id) -> None:
    template = template_cls()
    model = template.compile(template.default_spec())
    default_env = gym.make(env_id, max_episode_steps=50)
    morph_env = gym.make(
        env_id, xml_file=str(template.xml_path(template.default_spec())), max_episode_steps=50
    )
    try:
        assert (model.nq, model.nv, model.nu) == (
            default_env.unwrapped.model.nq,
            default_env.unwrapped.model.nv,
            default_env.unwrapped.model.nu,
        )
        assert morph_env.observation_space.shape == default_env.observation_space.shape
        assert morph_env.action_space.shape == default_env.action_space.shape
    finally:
        morph_env.close()
        default_env.close()


def test_hopper_geometry_coupling() -> None:
    template = HopperTemplate()
    values = template.defaults()
    values["thigh_len"] = 0.3
    long_thigh = template.compile(MorphologySpec.of(values))
    default = template.compile(template.default_spec())
    leg_id = default.body("leg").id
    assert long_thigh.body_pos[leg_id][2] < default.body_pos[leg_id][2]


def test_half_cheetah_geometry_coupling() -> None:
    template = HalfCheetahTemplate()
    values = template.defaults()
    values["torso_len"] = 0.6
    long_torso = template.compile(MorphologySpec.of(values))
    default = template.compile(template.default_spec())
    bthigh_id = default.body("bthigh").id
    fthigh_id = default.body("fthigh").id
    assert long_torso.body_pos[bthigh_id][0] < default.body_pos[bthigh_id][0]
    assert long_torso.body_pos[fthigh_id][0] > default.body_pos[fthigh_id][0]
    values = template.defaults()
    values["gear_scale"] = 1.5
    scaled = template.compile(MorphologySpec.of(values))
    assert scaled.actuator_gear[0][0] == pytest.approx(120.0 * 1.5)


def test_swimmer_parametric_coupling() -> None:
    template = SwimmerTemplate()
    values = template.defaults()
    values["torso_len"] = 0.6
    longer = template.compile(MorphologySpec.of(values))
    default = template.compile(template.default_spec())
    mid_id = default.body("mid").id
    assert longer.body_pos[mid_id][0] > default.body_pos[mid_id][0]


# --- Phase 2: topology-changing ---------------------------------------------------


def test_topology_templates_change_dimensions() -> None:
    swimmer = SwimmerTopologyTemplate()
    for n_links in (3, 4, 5, 6):
        spec = MorphologySpec.of({**swimmer.defaults(), "n_links": float(n_links)})
        model = swimmer.compile(spec)
        assert model.nu == n_links - 1
        assert model.nq == n_links + 2
    ant = AntTopologyTemplate()
    for n_legs in (4, 5, 6):
        spec = MorphologySpec.of({**ant.defaults(), "n_legs": float(n_legs)})
        model = ant.compile(spec)
        assert model.nu == 2 * n_legs
        assert model.nq == 2 * n_legs + 7


def test_count_field_validation_and_cost() -> None:
    swimmer = SwimmerTopologyTemplate()
    with pytest.raises(MorphologyError):
        swimmer.check(MorphologySpec.of({**swimmer.defaults(), "n_links": 3.5}))
    with pytest.raises(MorphologyError):
        swimmer.check(MorphologySpec.of({**swimmer.defaults(), "n_links": 2.0}))
    spec = MorphologySpec.of({**swimmer.defaults(), "n_links": 5.0})
    assert swimmer.validate(spec) == []
    assert "n_links 3->5" in swimmer.field_deltas(spec)
    assert morph_cost(swimmer, spec) > morph_cost(swimmer, swimmer.default_spec())


def test_ant_patterns_reproduce_default_4_leg_layout() -> None:
    from lawevo.pid.gym_benchmark import _ant_patterns

    patterns = _ant_patterns(8)
    np.testing.assert_array_equal(
        patterns["phase_offsets"], [0, 0, np.pi, np.pi, np.pi, np.pi, 0, 0]
    )
    np.testing.assert_array_equal(
        patterns["roll_pattern"], [1, 0.4, -1, -0.4, 1, 0.4, -1, -0.4]
    )
    np.testing.assert_array_equal(
        patterns["pitch_pattern"], [1, 0.4, 1, 0.4, -1, -0.4, -1, -0.4]
    )
    np.testing.assert_array_equal(patterns["height_pattern"], [1, 0.5] * 4)
    np.testing.assert_array_equal(
        patterns["speed_pattern"], [1, 0.5, -1, -0.5, -1, -0.5, 1, 0.5]
    )


def test_topology_adapters_run_with_dynamic_dims() -> None:
    swimmer = SwimmerTopologyTemplate()
    adapter = LOCOMOTION_ADAPTERS["swimmer"]
    spec = MorphologySpec.of({**swimmer.defaults(), "n_links": 4.0})
    env = make_morph_env(adapter, swimmer, spec)
    try:
        observation, _ = env.reset(seed=0)
        observation = adapter.prepare_reset(env, observation, 0)
        memory = adapter.reset_controller(3)
        features = adapter.features(env, observation, memory, env.unwrapped.dt)
        assert all(np.asarray(value).shape == (3,) for value in features.values())
        assert all(np.isfinite(value).all() for value in features.values())
    finally:
        env.close()
    gains, metrics, _ = tune_pair_cem(
        adapter, swimmer, spec, adapter.classical[2], [0, 1],
        iterations=1, population_size=4,
    )
    assert gains.shape == (2,)
    assert np.isfinite(metrics.score)

    ant = AntTopologyTemplate()
    ant_adapter = LOCOMOTION_ADAPTERS["ant"]
    spec_five = MorphologySpec.of({**ant.defaults(), "n_legs": 5.0})
    env = make_morph_env(ant_adapter, ant, spec_five)
    try:
        observation, _ = env.reset(seed=0)
        observation = ant_adapter.prepare_reset(env, observation, 0)
        memory = ant_adapter.reset_controller(10)
        features = ant_adapter.features(env, observation, memory, env.unwrapped.dt)
        assert all(np.asarray(value).shape == (10,) for value in features.values())
        assert all(np.isfinite(value).all() for value in features.values())
    finally:
        env.close()


def test_topology_runner_smoke() -> None:
    swimmer = SwimmerTopologyTemplate()
    adapter = LOCOMOTION_ADAPTERS["swimmer"]

    def law_generator(incumbent, belief, count, generation):
        pool = list(adapter.allowed_terms)
        return [
            GymStructure(
                f"top_{generation}_{index}",
                tuple(pool[(generation + index) % len(pool) :][:2]),
            )
            for index in range(count)
        ]

    def morph_generator(incumbent, belief, count, generation):
        return [
            MorphologySpec.of(
                {**swimmer.defaults(), "n_links": float(3 + ((generation + index) % 3))}
            )
            for index in range(count)
        ]

    config = MorpLawConfig(
        generations=2,
        proposals_per_side=2,
        joint_top_k=1,
        cem_iterations=1,
        cem_population=4,
        cross_direction="both",
    )
    runner = MorpLawRunner(adapter, swimmer, [0, 1], law_generator, morph_generator, config)
    best, reports = runner.run([(swimmer.default_spec(), adapter.classical[2])])
    assert len(reports) == 2
    assert np.isfinite(best.metrics.score)
    assert runner.belief.law_to_morph
