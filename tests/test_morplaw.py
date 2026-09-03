import json

import numpy as np
import pytest

pytest.importorskip("gymnasium", reason="MorpLaw tests need the benchmarks extras")
pytest.importorskip("mujoco")

import gymnasium as gym
import mujoco

from lawevo.evolve.belief import BeliefSpace, Experience
from lawevo.morplaw import (
    PRECISION_REACHER_ADAPTER,
    ROBOMORPH_ADAPTER,
    ROBOMORPH_TERRAINS,
    TEMPLATE_ADAPTERS,
    TEMPLATES,
    AntTemplate,
    AntTopologyTemplate,
    DirectedKnowledgeBase,
    EvidenceRecord,
    HalfCheetahTemplate,
    HopperTemplate,
    KnowledgeHypothesis,
    MorpLawConfig,
    MorpLawRunner,
    PairMetrics,
    PairRecord,
    PusherTemplate,
    ReacherGravityTemplate,
    ReacherPayloadTemplate,
    ReacherPrecisionTemplate,
    ReacherTemplate,
    RoboMorphGrammarTemplate,
    RobotGraphSpec,
    SwimmerTemplate,
    SwimmerTopologyTemplate,
    Walker2dTemplate,
    evaluate_pair,
    extract_law_proposals,
    extract_morphology_proposals,
    law_mutation_prompt,
    make_morph_env,
    morph_cost,
    morphology_mutation_prompt,
    tune_pair_cem,
)
from lawevo.morplaw.morphology import MorphologyError, MorphologySpec
from lawevo.morplaw.navigator import SearchDirective
from lawevo.pid.gym_benchmark import ADAPTERS, LOCOMOTION_ADAPTERS, GymStructure

REACHER = ADAPTERS["reacher"]


def _reacher_stub_law(prefix: str = "stub"):
    def law_generator(incumbent, belief, count, generation):
        terms_pool = list(REACHER.allowed_terms)
        return [
            GymStructure(
                f"{prefix}_{generation}_{index}",
                " + ".join(
                    f"K{i + 1}*{name}"
                    for i, name in enumerate(
                        terms_pool[(generation + index) % len(terms_pool) :][:2]
                    )
                ),
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
        "responsive_per_side": 0,
        "joint_top_k": 1,
        "cem_iterations": 1,
        "cem_population": 4,
    }
    base.update(overrides)
    return MorpLawConfig(**base)


def _run_reacher(config, template, law_prefix="stub", archive=None):
    law_gen = _reacher_stub_law(law_prefix)
    morph_gen = _reacher_stub_morph(template)
    runner = MorpLawRunner(REACHER, template, [0, 1], law_gen, morph_gen, config, archive=archive)
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


def test_pid_arm_variants_compile_with_distinct_dynamics() -> None:
    assert {
        key: TEMPLATE_ADAPTERS[key]
        for key in ("reacher_payload", "reacher_gravity", "reacher_precision", "pusher")
    } == {
        "reacher_payload": "reacher",
        "reacher_gravity": "reacher",
        "reacher_precision": "reacher_precision",
        "pusher": "pusher",
    }
    payload = ReacherPayloadTemplate()
    gravity = ReacherGravityTemplate()
    precision = ReacherPrecisionTemplate()
    base_spec = gravity.default_spec()

    payload_model = payload.compile(payload.default_spec())
    gravity_model = gravity.compile(base_spec)
    precision_model = precision.compile(precision.default_spec())
    assert payload_model.geom("payload").size[0] == pytest.approx(0.018)
    assert gravity_model.opt.gravity == pytest.approx((0.0, -9.81, 0.0))
    assert precision_model.geom("target").size[0] == pytest.approx(0.003)
    assert (
        len(
            {
                ReacherTemplate().xml_path(base_spec),
                gravity.xml_path(base_spec),
                precision.xml_path(base_spec),
            }
        )
        == 3
    )


@pytest.mark.parametrize(
    ("task_key", "adapter"),
    (
        ("reacher_payload", REACHER),
        ("reacher_gravity", REACHER),
        ("reacher_precision", PRECISION_REACHER_ADAPTER),
    ),
)
def test_pid_reacher_variants_reset_step_and_features(task_key, adapter) -> None:
    template = TEMPLATES[task_key]
    env = make_morph_env(adapter, template, template.default_spec())
    try:
        observation, _ = env.reset(seed=0)
        observation = adapter.prepare_reset(env, observation, 0)
        action_dim = int(np.prod(env.action_space.shape))
        features = adapter.features(
            env,
            observation,
            adapter.reset_controller(action_dim),
            env.unwrapped.dt,
        )
        assert set(features) == set(adapter.allowed_terms)
        assert all(np.asarray(value).shape == (action_dim,) for value in features.values())
        observation, reward, _, _, _ = env.step(np.zeros(action_dim, dtype=np.float32))
        assert np.all(np.isfinite(observation))
        assert np.isfinite(reward)
    finally:
        env.close()


def test_morphable_pusher_geometry_adapter_and_rollout() -> None:
    template = PusherTemplate()
    default = template.compile(template.default_spec())
    values = template.defaults()
    values.update({"upper_len": 0.5, "forearm_len": 0.35, "gear": 1.5})
    changed = template.compile(MorphologySpec.of(values))
    assert default.nu == changed.nu == 7
    assert changed.body("r_elbow_flex_link").pos[0] == pytest.approx(0.5)
    assert changed.body("r_wrist_flex_link").pos[0] == pytest.approx(0.38)
    assert changed.actuator_gear[:, 0] == pytest.approx(np.full(7, 1.5))

    adapter = ADAPTERS["pusher"]
    env = make_morph_env(adapter, template, template.default_spec())
    try:
        observation, _ = env.reset(seed=0)
        observation = adapter.prepare_reset(env, observation, 0)
        memory = adapter.reset_controller(7)
        features = adapter.features(env, observation, memory, env.unwrapped.dt)
        assert set(features) == set(adapter.allowed_terms)
        assert all(np.asarray(value).shape == (7,) for value in features.values())
        observation, reward, _, _, _ = env.step(np.zeros(7, dtype=np.float32))
        assert np.all(np.isfinite(observation))
        assert np.isfinite(reward)
    finally:
        env.close()


def test_pid_arm_prompts_have_complete_task_specific_context() -> None:
    adapters = {
        "reacher_payload": REACHER,
        "reacher_gravity": REACHER,
        "reacher_precision": PRECISION_REACHER_ADAPTER,
        "pusher": ADAPTERS["pusher"],
    }
    directive = SearchDirective(1, "explore", "test", "vary law", "vary body", "pair them")
    knowledge = DirectedKnowledgeBase("full")
    metrics = PairMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1, 0.0, 1.0)
    for task_key, adapter in adapters.items():
        template = TEMPLATES[task_key]
        structure = adapter.classical[0]
        incumbent = PairRecord(
            template.default_spec(),
            structure,
            np.zeros(structure.parameter_count),
            metrics,
            0,
        )
        law_prompt = law_mutation_prompt(
            task_key,
            adapter,
            incumbent,
            knowledge,
            [],
            directive,
            [],
            1,
            1,
        )
        morph_prompt = morphology_mutation_prompt(
            task_key,
            template,
            incumbent,
            knowledge,
            [],
            directive,
            [],
            1,
            1,
        )
        assert "TASK AND ENVIRONMENT" in law_prompt
        assert "MORPHOLOGY / TOPOLOGY PHYSICS" in morph_prompt


def test_pid_arm_pair_cache_is_task_scoped(tmp_path) -> None:
    from experiments.evolve_morplaw import load_cache, save_cache

    template = TEMPLATES["reacher_payload"]
    structure = REACHER.classical[0]
    record = PairRecord(
        template.default_spec(),
        structure,
        np.zeros(structure.parameter_count),
        PairMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1, 0.0, 1.0),
        0,
    )
    path = tmp_path / "records.jsonl"
    save_cache(path, {record.key(): record}, "reacher_payload", "Reacher-v5")
    assert load_cache(path, "reacher_payload", "Reacher-v5")
    assert load_cache(path, "reacher_gravity", "Reacher-v5") == {}


def test_robomorph_grammar_seeds_are_diverse_valid_graphs() -> None:
    template = RoboMorphGrammarTemplate()
    seeds = template.seed_specs(4, seed=7)
    assert len(seeds) == 4
    assert len({spec.key() for spec in seeds}) == 4
    renamed = RobotGraphSpec(seeds[0].body, "cosmetic_rename")
    assert renamed.key() == seeds[0].key()
    assert template.xml_path(renamed) == template.xml_path(seeds[0])
    for spec in seeds:
        assert isinstance(spec, RobotGraphSpec)
        assert template.validate(spec) == []
        model = template.compile(spec)
        assert model.nu == spec.counts()["actuators"]
        assert 2 <= model.nu <= 16


def test_robomorph_terrain_registry_geometry_and_cache_isolation() -> None:
    expected = {f"robomorph_{terrain}" for terrain in ROBOMORPH_TERRAINS}
    assert expected <= TEMPLATES.keys()
    assert {TEMPLATE_ADAPTERS[name] for name in expected} == {"robomorph"}

    spec = RoboMorphGrammarTemplate().default_spec()
    paths = {
        terrain: RoboMorphGrammarTemplate(terrain).xml_path(spec) for terrain in ROBOMORPH_TERRAINS
    }
    assert len(set(paths.values())) == len(ROBOMORPH_TERRAINS)

    models = {
        terrain: RoboMorphGrammarTemplate(terrain).compile(spec) for terrain in ROBOMORPH_TERRAINS
    }
    assert models["flat"].geom("floor").friction[0] == pytest.approx(1.2)
    assert models["frozen_lake"].geom("floor").friction[0] == pytest.approx(0.05)
    assert mujoco.mj_name2id(models["flat"], mujoco.mjtObj.mjOBJ_GEOM, "ridge_0") == -1
    assert mujoco.mj_name2id(models["frozen_lake"], mujoco.mjtObj.mjOBJ_GEOM, "beam_0") == -1

    for terrain, prefix, height in (("ridged", "ridge", 0.0), ("beams", "beam", 0.5)):
        first = models[terrain].geom(f"{prefix}_0")
        last = models[terrain].geom(f"{prefix}_14")
        assert first.pos == pytest.approx((1.0, 0.0, height))
        assert last.pos == pytest.approx((29.0, 0.0, height))
        assert first.size[:2] == pytest.approx((0.2, 10.0))


@pytest.mark.parametrize("terrain", ROBOMORPH_TERRAINS)
def test_robomorph_terrain_env_resets_and_steps(terrain: str) -> None:
    template = RoboMorphGrammarTemplate(terrain)
    env = make_morph_env(ROBOMORPH_ADAPTER, template, template.default_spec())
    try:
        observation, _ = env.reset(seed=0)
        assert np.all(np.isfinite(observation))
        assert env.unwrapped._ctrl_cost_weight == pytest.approx(0.0)
        assert env.unwrapped._healthy_reward == pytest.approx(0.0)
        observation, reward, terminated, truncated, _ = env.step(
            np.zeros(env.action_space.shape, dtype=np.float32)
        )
        assert np.all(np.isfinite(observation))
        assert np.isfinite(reward)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
    finally:
        env.close()


def test_robomorph_graph_parser_compiles_bilateral_passive_wheels() -> None:
    template = RoboMorphGrammarTemplate()
    response = json.dumps(
        [
            {
                "name": "wheeled_biped",
                "graph": {
                    "body": [
                        {
                            "length": 0.22,
                            "joint_to_previous": "root",
                            "limb": {
                                "segments": [{"joint": "knee", "length": 0.18, "angle": 20}],
                                "terminal": "wheel",
                            },
                        }
                    ]
                },
                "knowledge": {
                    "summary": "passive rolling may reduce tangential contact loss",
                    "recommendation": "test wheels below actuated knees",
                    "condition": "flat terrain",
                    "prediction": {"score": "increase"},
                },
            }
        ]
    )
    proposals = extract_morphology_proposals(response, template)
    assert len(proposals) == 1
    spec = proposals[0].spec
    assert isinstance(spec, RobotGraphSpec)
    assert spec.counts() == {
        "body_segments": 1,
        "limb_pairs": 1,
        "limb_segments": 2,
        "wheels": 2,
        "actuators": 2,
    }
    model = template.compile(spec)
    assert model.nu == 2
    assert model.njnt == 5  # free root + 2 actuated knees + 2 passive wheels
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "limb_b0_l_0") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "limb_b0_r_0") >= 0
    invalid = response.replace('"knee"', '"rigid"')
    assert extract_morphology_proposals(invalid, template) == []


def test_robomorph_adapter_tracks_live_actuator_graph() -> None:
    template = RoboMorphGrammarTemplate()
    spec = template.seed_specs(2, seed=11)[1]
    env = make_morph_env(ROBOMORPH_ADAPTER, template, spec)
    try:
        observation, _ = env.reset(seed=0)
        observation = ROBOMORPH_ADAPTER.prepare_reset(env, observation, 0)
        action_dim = int(np.prod(env.action_space.shape))
        memory = ROBOMORPH_ADAPTER.reset_controller(action_dim)
        features = ROBOMORPH_ADAPTER.features(env, observation, memory, env.unwrapped.dt)
        assert set(features) == set(ROBOMORPH_ADAPTER.allowed_terms)
        assert all(np.asarray(value).shape == (action_dim,) for value in features.values())
        env.step(np.zeros(action_dim, dtype=np.float32))
    finally:
        env.close()


def test_morplaw_engine_evolves_complete_robot_graphs(monkeypatch) -> None:
    template = RoboMorphGrammarTemplate()
    candidate = RobotGraphSpec.from_dict(
        {
            "name": "three_body_wheeler",
            "body": [
                {
                    "length": 0.2,
                    "limb": {
                        "segments": [{"joint": "knee", "length": 0.16}],
                        "terminal": "wheel",
                    },
                },
                {"length": 0.18, "joint_to_previous": "roll", "limb": None},
                {
                    "length": 0.2,
                    "joint_to_previous": "rigid",
                    "limb": {
                        "segments": [{"joint": "knee", "length": 0.16}],
                        "terminal": "wheel",
                    },
                },
            ],
        }
    )
    template.check(candidate)

    def fake_tune(adapter, body_template, spec, structure, seeds, **kwargs):
        del adapter, body_template, seeds, kwargs
        counts = spec.counts()
        score = 10.0 * counts["body_segments"] + counts["wheels"] + structure.parameter_count
        metrics = PairMetrics(score, score, 1.0, 1.0, 0.1, 0.1, structure.parameter_count, 0.1, 5.0)
        return np.zeros(structure.parameter_count), metrics, 1

    monkeypatch.setattr("lawevo.morplaw.engine.tune_pair_cem", fake_tune)

    def laws(incumbent, knowledge, count, generation):
        del incumbent, knowledge, generation
        return [
            GymStructure("graph_pd", "K1*phase_sin + K2*posture_error + K3*joint_velocity")
            for _ in range(count)
        ]

    def morphs(incumbent, knowledge, count, generation):
        del incumbent, knowledge, generation
        return [candidate for _ in range(count)]

    runner = MorpLawRunner(
        ROBOMORPH_ADAPTER,
        template,
        [0],
        laws,
        morphs,
        _small_config(
            generations=1,
            proposals_per_side=1,
            responsive_per_side=0,
            knowledge_mode="full",
        ),
    )
    best, reports = runner.run([(template.default_spec(), ROBOMORPH_ADAPTER.classical[2])])
    assert best.spec.key() == candidate.key()
    assert reports[0].cross_table["morph_cross"] == 1
    assert any(
        item.hypothesis.direction == "law_to_morph" for item in runner.knowledge.items.values()
    )


def test_equal_cem_budget_across_pairs() -> None:
    template = ReacherTemplate()
    structure = REACHER.classical[0]
    _, _, budget_default = tune_pair_cem(
        REACHER,
        template,
        template.default_spec(),
        structure,
        [0, 1],
        iterations=1,
        population_size=4,
    )
    values = template.defaults()
    values["gear"] = 250.0
    _, _, budget_changed = tune_pair_cem(
        REACHER,
        template,
        MorphologySpec.of(values),
        structure,
        [0, 1],
        iterations=1,
        population_size=4,
    )
    assert budget_default == budget_changed == 2 * (1 + 1 * 4)


# --- runner ------------------------------------------------------------------


def test_runner_elitism_determinism_and_archive_dedup() -> None:
    template = ReacherTemplate()
    config = _small_config(knowledge_mode="full")
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


def test_direction_gating_and_evidence_is_ablation_invariant() -> None:
    template = ReacherTemplate()
    evidence_counts = []
    for direction, morph_nonempty, law_nonempty in [
        ("no_knowledge", False, False),
        ("m_to_l", True, False),
        ("l_to_m", False, True),
        ("full", True, True),
    ]:
        runner, _ = _run_reacher(_small_config(knowledge_mode=direction), template)
        morph_items = [
            item
            for item in runner.knowledge.items.values()
            if item.hypothesis.direction == "morph_to_law"
        ]
        law_items = [
            item
            for item in runner.knowledge.items.values()
            if item.hypothesis.direction == "law_to_morph"
        ]
        assert bool(morph_items) == morph_nonempty, direction
        assert bool(law_items) == law_nonempty, direction
        evidence_counts.append(len(runner.knowledge.evidence))
    assert len(set(evidence_counts)) == 1


def test_archive_dedup_skips_reevaluation() -> None:
    template = ReacherTemplate()
    structure = REACHER.classical[0]

    def constant_law(incumbent, belief, count, generation):
        return [structure.renamed("constant") for _ in range(count)]

    def constant_morph(incumbent, belief, count, generation):
        return [template.default_spec() for _ in range(count)]

    config = _small_config(knowledge_mode="no_knowledge")
    runner = MorpLawRunner(REACHER, template, [0, 1], constant_law, constant_morph, config)
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


def test_shared_evaluation_cache_does_not_leak_search_archive() -> None:
    template = ReacherTemplate()
    initial = [(template.default_spec(), structure) for structure in REACHER.classical[:2]]
    evaluation_cache = {}
    first_archive = {}
    first = MorpLawRunner(
        REACHER,
        template,
        [0, 1],
        _reacher_stub_law(),
        _reacher_stub_morph(template),
        _small_config(generations=1),
        archive=first_archive,
        evaluation_cache=evaluation_cache,
    )
    first.run(initial)
    evolved_keys = set(evaluation_cache) - {
        (structure.key(), template.default_spec().key()) for _, structure in initial
    }
    assert evolved_keys

    second_archive = {}
    second = MorpLawRunner(
        REACHER,
        template,
        [0, 1],
        _reacher_stub_law(),
        _reacher_stub_morph(template),
        _small_config(generations=0),
        archive=second_archive,
        evaluation_cache=evaluation_cache,
    )
    second.run(initial)
    assert not (set(second.archive) & evolved_keys)
    assert second.episodes_spent == 0


def test_frozen_flags_produce_single_side_runs() -> None:
    template = ReacherTemplate()
    runner, (best, _) = _run_reacher(_small_config(morphology_frozen=True), template)
    assert all(
        record.spec.key() == template.default_spec().key() for record in runner.archive.values()
    )
    assert not any(
        item.hypothesis.direction == "law_to_morph" for item in runner.knowledge.items.values()
    )
    assert runner.calls == {"law": 2, "morph": 0}
    assert best.spec.key() == template.default_spec().key()

    runner, _ = _run_reacher(_small_config(law_frozen=True), template)
    assert not any(
        item.hypothesis.direction == "morph_to_law" for item in runner.knowledge.items.values()
    )
    assert runner.calls == {"law": 0, "morph": 2}


def test_llm_call_and_requested_episode_parity_across_ablations() -> None:
    template = ReacherTemplate()
    requested = []
    for direction in ("full", "m_to_l", "l_to_m", "no_knowledge"):
        runner, _ = _run_reacher(
            _small_config(
                generations=1,
                responsive_per_side=1,
                knowledge_mode=direction,
            ),
            template,
        )
        assert runner.calls == {"law": 2, "morph": 2}, direction
        requested.append(runner.episodes_requested)
    assert len(set(requested)) == 1


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


def test_directed_knowledge_keeps_positive_and_negative_with_soft_retrieval() -> None:
    knowledge = DirectedKnowledgeBase("full", capacity_per_bank=4)
    positive = KnowledgeHypothesis(
        "morph_to_law",
        "Higher gear supports bounded error feedback.",
        "use tanh error with task damping",
        "gear is high relative to link mass",
        {"score": "increase"},
        "add task damping",
    )
    negative = KnowledgeHypothesis(
        "morph_to_law",
        "Integral action oscillates on this body family.",
        "add integral error",
        "links are light and actuator gear is high",
        {"jerk": "increase"},
        "add integral error",
    )
    context = {
        "task": "Reacher-v5",
        "morphology": {"gear": 200.0, "l0": 0.1},
        "law_terms": ["jt_error", "task_damping"],
    }
    knowledge.observe(
        positive,
        EvidenceRecord(
            "morph_to_law",
            1,
            "parent",
            "positive-child",
            "add task damping",
            0.5,
            {"score": 0.5},
            context,
            positive.id,
        ),
    )
    knowledge.observe(
        negative,
        EvidenceRecord(
            "morph_to_law",
            1,
            "parent",
            "negative-child",
            "add integral error",
            -0.4,
            {"score": -0.4},
            context,
            negative.id,
        ),
    )
    retrieved = knowledge.retrieve(
        "morph_to_law",
        {
            "task": "Reacher-v5",
            "morphology": {"gear": 210.0, "l0": 0.095},
            "law_terms": ["jt_error", "task_damping"],
        },
        generation=2,
    )
    assert {item.polarity for item in retrieved} == {"positive", "negative"}
    assert "insights_to_follow" in knowledge.summary(retrieved)
    assert "pitfalls_to_avoid" in knowledge.summary(retrieved)


def test_knowledge_first_proposal_parsers_preserve_hypotheses() -> None:
    laws = extract_law_proposals(
        """[{"name":"bounded_pd","expression":"K1*tanh(jt_error) + K2*task_damping",
        "knowledge":{"summary":"bounded error needs damping",
        "recommendation":"pair bounded error with task damping",
        "condition":"light high-gear links",
        "prediction":{"score":"increase","jerk":"decrease"}}}]""",
        REACHER.allowed_terms,
        retrieved_ids=("kh-old:positive",),
    )
    assert laws[0].hypothesis.direction == "morph_to_law"
    assert laws[0].hypothesis.prediction["jerk"] == "decrease"
    assert laws[0].retrieved_ids == ("kh-old:positive",)

    template = ReacherTemplate()
    values = template.defaults()
    values["gear"] = 240.0
    response = (
        '[{"values":' + json.dumps(values) + ',"knowledge":{"summary":"more torque margin",'
        '"recommendation":"raise gear","condition":"action saturation",'
        '"prediction":{"score":"increase"}}}]'
    )
    morphs = extract_morphology_proposals(response, template)
    assert morphs[0].spec.get("gear") == 240.0
    assert morphs[0].hypothesis.direction == "law_to_morph"


def test_morphology_delta_is_parent_relative() -> None:
    template = ReacherTemplate()
    parent_values = template.defaults()
    parent_values["gear"] = 240.0
    child_values = dict(parent_values)
    child_values["gear"] = 260.0
    delta = template.field_deltas(MorphologySpec.of(child_values), MorphologySpec.of(parent_values))
    assert "gear 240->260" in delta
    assert "gear 200->260" not in delta


def test_responsive_factorial_interactions_and_call_budget() -> None:
    template = ReacherTemplate()
    runner, (_, reports) = _run_reacher(
        _small_config(
            generations=1,
            responsive_per_side=1,
            knowledge_mode="full",
        ),
        template,
    )
    assert runner.calls == {"law": 2, "morph": 2}
    assert len(reports[0].interactions) == 3
    assert reports[0].cross_table["responsive"] == 4
    for item in reports[0].interactions:
        expected = item.joint_score - item.morph_score - item.law_score + item.baseline_score
        assert item.interaction == pytest.approx(expected)


# --- pair evaluation -----------------------------------------------------------


def test_evaluate_pair_returns_normalized_metrics() -> None:
    template = ReacherTemplate()
    spec = template.default_spec()
    structure = REACHER.classical[0]
    gains, train_metrics, _ = tune_pair_cem(
        REACHER, template, spec, structure, [0, 1], iterations=1, population_size=4
    )
    metrics, episodes = evaluate_pair(REACHER, template, spec, structure, gains, [1000, 1001])
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
    np.testing.assert_array_equal(patterns["roll_pattern"], [1, 0.4, -1, -0.4, 1, 0.4, -1, -0.4])
    np.testing.assert_array_equal(patterns["pitch_pattern"], [1, 0.4, 1, 0.4, -1, -0.4, -1, -0.4])
    np.testing.assert_array_equal(patterns["height_pattern"], [1, 0.5] * 4)
    np.testing.assert_array_equal(patterns["speed_pattern"], [1, 0.5, -1, -0.5, -1, -0.5, 1, 0.5])


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
        adapter,
        swimmer,
        spec,
        adapter.classical[2],
        [0, 1],
        iterations=1,
        population_size=4,
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
                " + ".join(
                    f"K{i + 1}*{name}"
                    for i, name in enumerate(
                        pool[(generation + index) % len(pool) :][:2]
                    )
                ),
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
        responsive_per_side=0,
        joint_top_k=1,
        cem_iterations=1,
        cem_population=4,
        knowledge_mode="full",
    )
    runner = MorpLawRunner(adapter, swimmer, [0, 1], law_generator, morph_generator, config)
    best, reports = runner.run([(swimmer.default_spec(), adapter.classical[2])])
    assert len(reports) == 2
    assert np.isfinite(best.metrics.score)
    assert any(
        item.hypothesis.direction == "law_to_morph" for item in runner.knowledge.items.values()
    )
