"""Natural-language task and morphology knowledge for MorpLaw prompts.

Descriptions are physics-first and intentionally do not mention any benchmark
framework: the LLM should reason about the robot body and its dynamics, not an
environment id. Keys are the MorpLaw template names.
"""

TASK_DESCRIPTIONS = {
    "reacher": """A planar robot arm is fixed to a base. Link 1 attaches to the base through a
shoulder hinge and link 2 attaches to link 1 through an elbow hinge; both hinges rotate in
the horizontal plane, so gravity does not load the joints — only link inertia and joint
damping resist motion. One torque motor drives each hinge; the controller command for each
motor is a normalized value in [-1, 1] scaled by an actuator gear. A small fingertip sits at
the end of link 2, and a target point is placed at a random reachable position in the plane.
The episode lasts 50 control steps. The goal is to bring the fingertip onto the target and
keep it there: final Cartesian fingertip-target distance below 0.05 counts as success, and
the return improves the earlier and the closer the fingertip approaches. Longer links reach
farther but carry more inertia, so they accelerate more slowly for the same motor torque;
heavier (denser) links and weaker gears slow the arm further. The joint angles, joint
velocities, and fingertip-to-target offset are observable.""",
    "walker2d": """A planar bipedal robot stands on a flat floor. Its rigid torso carries two
identical legs, each made of three segments: a thigh attached to the torso by a hip hinge, a
shin attached to the thigh by a knee hinge, and a foot attached to the shin by an ankle
hinge. Six torque motors (one per hinge) drive the legs; each motor command is a normalized
value in [-1, 1] scaled by an actuator gear. The torso can slide forward/upward and pitch in
the vertical plane. The episode lasts 300 control steps and ends early if the torso height
drops below about 0.7 m or the torso pitch leaves roughly [-0.2, 0.2] rad (a fall). The goal
is sustained forward walking: survival for the full horizon and a final forward speed above
0.75 m/s. Walking is driven by alternating the two legs; symmetric simultaneous feedback to
both legs tends to suppress the alternating gait. The measurable signals include each joint's
posture error and velocity, torso height and pitch error, forward-speed error, and a
fixed-frequency anti-phase oscillation (CPG) between the legs. Training evaluations randomly
perturb body masses by +/-10%, so gaits must not depend on exact mass values.""",
    "hopper": """A planar one-legged hopping robot stands on a flat floor. A rigid torso sits
on a single leg with three segments — thigh, leg, and foot — connected by three hinge joints,
each driven by a torque motor with a normalized command in [-1, 1] scaled by an actuator
gear. The torso can slide forward/upward and pitch. The episode lasts 300 control steps and
ends early if the torso height drops below about 0.7 m or the torso pitch leaves roughly
[-0.2, 0.2] rad. The goal is to hop forward without falling: survive the full horizon, keep
torso height near 1.25 m, and end with forward speed above 0.75 m/s. Propulsion comes from a
periodic hop cycle created by the three leg joints; posture, height, and pitch feedback must
keep the body upright between contacts. Training evaluations randomly perturb body masses by
+/-10%, so the gait must tolerate mass variation.""",
    "half_cheetah": """A planar running robot has a long rigid torso (a capsule) with a small
head at the front, and two identical three-segment legs attached to its underside: a rear
leg behind the torso center and a front leg ahead of it. Each leg is a chain of thigh, shin,
and foot segments connected by hinge joints; six torque motors in total drive them, with
normalized commands in [-1, 1] scaled by per-motor gears. The torso can slide forward/upward
and pitch in the vertical plane. The episode lasts 300 control steps and does not terminate
on a fall, but the reward strongly favors forward velocity. The goal is fast sustained
running: develop a cyclic propulsive gait and end above 1.5 m/s rather than holding a static
posture or making one initial kick. Front and rear legs must be coordinated; high-frequency
torque chatter wastes effort. Measurable signals include per-joint posture errors and
velocities, body pitch, torso height, forward-speed error, and a fixed-frequency anti-phase
oscillation between front and rear. Training evaluations randomly perturb body masses by
+/-10%.""",
    "swimmer": """A planar three-link swimmer moves through viscous fluid. A head link carries
two trailing links behind it; each trailing link attaches to the previous one through a hinge
joint, and two torque motors (normalized commands in [-1, 1] scaled by a gear) drive those
two hinges. The head is free to slide in the plane and rotate, and the fluid applies strong
drag proportional to velocity. The episode lasts 300 control steps. The goal is forward
propulsion: travel at least 1.0 m forward by generating a coordinated traveling body wave —
in viscous fluid a symmetric back-and-forth stroke or a static pose produces almost no net
displacement, so the wave must be asymmetric in time (non-reciprocal). Measurable signals
include the two joint angles and velocities, body orientation, lateral velocity,
forward-speed error, and a fixed-frequency oscillation. Training evaluations randomly perturb
body masses by +/-10%.""",
    "ant": """A three-dimensional quadruped robot has a spherical torso and four identical
legs, each leg made of two segments: a short upper segment that reaches from the torso to a
hip joint, and a lower segment that reaches from the hip to an ankle joint. Each leg has two
hinges: the hip rotates the leg about the vertical axis and the ankle swings the lower
segment, so eight torque motors in total drive the robot, with normalized commands in [-1, 1]
scaled by a gear. The torso is a free body in 3D (it can translate, pitch, roll, and yaw).
The episode lasts 300 control steps and ends early if the body becomes unhealthy (torso
falls). The goal is to walk forward: survive the full horizon, keep torso height near 0.65 m
with controlled roll and pitch, and end above 0.75 m/s. Gaits coordinate legs in diagonal
pairs. Longer legs raise the body and stride but increase leg inertia and the height of the
center of mass; a heavier torso relative to the legs stabilizes the body; stronger gears
give more leg authority. Training evaluations randomly perturb body masses by +/-10%.""",
    "swimmer_topology": """A planar multi-link swimmer moves through viscous fluid. A head
link carries a chain of trailing links behind it; each trailing link attaches to the previous
one through a hinge joint, and one torque motor per hinge (normalized command in [-1, 1]
scaled by a gear) drives it. The head is free to slide in the plane and rotate, and the fluid
applies strong drag proportional to velocity. The episode lasts 300 control steps. The goal
is forward propulsion: travel at least 1.0 m forward by generating a coordinated traveling
body wave — in viscous fluid a symmetric back-and-forth stroke or a static pose produces
almost no net displacement. The number of links is a design variable: more links give a
smoother, larger-amplitude body wave and more actuators, but the chain is heavier, harder to
coordinate, and each extra link adds one hinge motor and increases the action dimension (the
controller has one scalar gain per term and applies to any number of actuators). Measurable
signals include each joint angle and velocity, body orientation, lateral velocity,
forward-speed error, and a fixed-frequency oscillation.""",
    "ant_topology": """A three-dimensional multi-legged robot has a spherical torso and
several identical legs attached around it, spread evenly in the horizontal plane. Each leg
has two segments: a short upper segment from the torso to a hip joint and a lower segment
from the hip to an ankle joint. Each leg has two hinges — the hip rotates the leg about the
vertical axis and the ankle swings the lower segment — and one torque motor per hinge, with
normalized commands in [-1, 1] scaled by a gear. The torso is a free body in 3D. The episode
lasts 300 control steps and ends early if the body becomes unhealthy (torso falls). The goal
is to walk forward: survive the full horizon, keep torso height near 0.65 m with controlled
roll and pitch, and end above 0.75 m/s. The number of legs is a design variable: more legs
distribute support and add actuators (two per leg, increasing the action dimension — the
controller has one scalar gain per term and applies to any leg count), but they add weight
and coordination burden. Gaits coordinate legs in diagonal pairs. Longer legs raise the body
and stride but increase leg inertia and center-of-mass height; stronger gears give more leg
authority. Training evaluations randomly perturb body masses by +/-10%.""",
}

CONTROL_GOALS = {
    "reacher": """Primary goal: move the fingertip onto the target quickly and keep it there,
maximizing return over the 50-step episode; final fingertip-target distance below 0.05
counts as success across randomized targets and link masses. Near the target, suppress
overshoot, oscillation, and torque chatter. Among structures with comparable accuracy,
prefer lower torque energy, smoother torque changes, and fewer terms. A fast initial swing
that cannot settle is not success. When designing for THIS body, remember the trade-off the
morphology imposes: long heavy links need strong low-frequency feedback and damping to stop
cleanly, while short light links can use more aggressive proportional action.""",
    "walker2d": """Primary goal: walk forward quickly and survive all 300 steps; success
requires surviving without a fall and ending above 0.75 m/s. Maintain torso height near
1.25 m and pitch near zero while producing an alternating-leg gait. Prefer speed achieved
with lower torque energy and smoother commands, and reject solutions that stand still or
fall early. For THIS body, the law must match the leg geometry: longer legs swing slower
(the CPG and posture gains that work for short legs will mistime long ones), heavier legs
need more hip/knee authority, and stronger gears allow smaller gains for the same motion.""",
    "hopper": """Primary goal: hop forward quickly and remain healthy for all 300 steps;
success requires surviving without a fall and ending above 0.75 m/s. Keep torso height near
1.25 m and pitch near zero while coordinating a repeatable hop cycle. Prefer gaits with
lower torque energy and smoother commands; do not buy speed with violent impacts that fail
under +/-10% mass variation. For THIS body, the hop timing depends on leg length and mass:
longer legs take longer strides but swing more slowly, heavier legs require more drive, and
gear strength sets how quickly the controller can inject energy into the hop.""",
    "half_cheetah": """Primary goal: maximize forward-running return over 300 steps and end
above 1.5 m/s. Develop a stable cyclic gait rather than a static posture or a single kick.
Prefer lower torque energy, lower torque-rate jerk, and fewer signals once speed is
comparable. For THIS body, front/rear coordination must respect the torso geometry: a
longer torso spaces the legs farther apart (leg attachment points move with torso length),
so the phase relationship between front and rear legs must change with it; longer segments
increase limb inertia, and the gear scale sets the whole actuator budget.""",
    "swimmer": """Primary goal: generate sustained forward propulsion for 300 steps and move
at least 1.0 m from the start. Discover a repeatable traveling body wave, not a static pose
or one impulse. Minimize torque energy, jerk, lateral drift, and unnecessary terms among
comparable swimmers. For THIS body, the wave shape must match the segment lengths: longer
segments propagate the wave over more body length and need appropriately scaled joint
amplitudes; heavier/denser segments and weaker gears slow the wave frequency.""",
    "ant": """Primary goal: walk forward quickly, stay healthy for all 300 steps, and end
above 0.75 m/s. Maintain torso height near 0.65 m and control roll/pitch with coordinated
diagonal-leg cycles. Minimize torque energy and jerk among equally successful gaits; do not
trade robustness for impact-heavy hopping. For THIS body, the gait must respect the leg
geometry: longer legs raise the body and need different contact timing, leg mass relative to
the torso changes how much the legs can swing the body, and gear strength sets joint
authority.""",
    "swimmer_topology": """Primary goal: generate sustained forward propulsion for 300 steps
and move at least 1.0 m from the start, for any proposed number of links. Discover a
repeatable traveling body wave. Minimize torque energy, jerk, lateral drift, and unnecessary
terms among comparable swimmers. Because the law has one scalar gain per term and applies to
any actuator count, prefer structures whose signals are naturally periodic and distributed
(sinusoidal phases, posture, damping) over anything that assumes a fixed joint count. For
THIS body, the wave must match the number and length of segments: more links allow a
smoother wave but require the phases to advance evenly along the chain.""",
    "ant_topology": """Primary goal: walk forward quickly, stay healthy for all 300 steps,
and end above 0.75 m/s, for any proposed number of legs. Maintain torso height near 0.65 m
and control roll/pitch with coordinated diagonal-leg cycles. Minimize torque energy and jerk
among equally successful gaits. Because the law has one scalar gain per term and applies to
any actuator count, prefer structures whose signals are naturally periodic and distributed
(sinusoidal phases, posture, damping) over anything that assumes a fixed leg count. For THIS
body, gaits must respect the leg count and geometry: with more legs, diagonal pairing and
phase offsets must scale accordingly.""",
}

TERM_SEMANTICS = {
    "reacher": """- jt_error: the two-vector J^T e, where J is the arm Jacobian and e is the
Cartesian fingertip-to-target error — the natural proportional task-space feedback.
- joint_velocity: the two joint angular velocities — damping.
- integral_jt_error: clipped time integral of jt_error (max magnitude 0.5) — removes
persistent target offset but can wind up.
- tanh_jt_error: tanh(10 * jt_error) — strong bounded feedback, saturates far from target.
- tanh_velocity: tanh(joint_velocity) — bounded damping.
- normalized_jt_error: jt_error divided by its own norm — unit-magnitude direction signal;
tends to chatter near the target.
- task_damping: J^T J qdot — damps Cartesian (task-space) motion rather than each joint.""",
    "locomotion": """All terms are vectors with one component per actuator.
- phase_sin / phase_cos: a fixed-frequency traveling oscillation (CPG) whose per-actuator
phase advances along the body — generates the cyclic motion; both together set amplitude and
offset.
- posture_error: the negative of each joint position — pulls joints back to the reference
pose; too strong freezes the gait, too weak lets limbs collapse.
- joint_velocity: joint angular velocities — damping against chatter and impact.
- integral_posture: clipped time integral of posture error — removes persistent posture
offset but can wind up.
- tanh_posture: tanh(2 * posture_error) — bounded posture feedback.
- tanh_velocity: tanh(joint_velocity) — bounded damping.
- body_angle: torso roll/pitch (or body orientation) weighted per actuator — attitude
correction; keeps the torso upright.
- height_error: the torso-height deficit weighted per actuator — prevents collapse.
- forward_speed_error: the target-speed deficit weighted per actuator — accelerates the
gait.""",
    "swimmer": """All terms are vectors with one component per actuator.
- phase_sin / phase_cos: a fixed-frequency oscillation whose per-actuator phase alternates
along the chain — the body wave; a traveling wave needs both quadrature components.
- posture_error: the negative of each joint angle — straightens the chain.
- joint_velocity: joint angular velocities — damping.
- integral_posture: clipped integral of posture error.
- tanh_posture / tanh_velocity: bounded versions of posture and damping.
- body_angle: the head orientation weighted with alternating signs along the chain — steers
the wave.
- lateral_velocity: sideways velocity weighted with alternating signs — cancels drift.
- forward_speed_error: the forward-speed deficit applied to every actuator.""",
    "ant": """All terms are vectors with one component per actuator (hip then ankle of each
leg).
- phase_sin / phase_cos: a fixed-frequency oscillation whose per-actuator phase realizes the
diagonal-leg pattern — the gait generator.
- posture_error: the negative of each joint position — holds the reference pose.
- joint_velocity: joint angular velocities — damping.
- integral_posture: clipped integral of posture error.
- tanh_posture / tanh_velocity: bounded posture and damping.
- body_angle: torso roll and pitch weighted per actuator — attitude correction.
- height_error: torso-height deficit weighted per actuator — prevents collapse.
- forward_speed_error: target-speed deficit weighted per actuator — accelerates the gait.""",
}

MORPHOLOGY_GUIDANCE = {
    "reacher": """Field physics (l0, l1 = link lengths; r0, r1 = link radii; density0,
density1 = link material density; gear = motor strength multiplier):
- Longer links increase reach so distant targets become reachable, but they raise link
inertia, so the same motor torque produces slower acceleration, and the same damping
feedback must work harder to stop cleanly.
- Denser links add mass without adding length: more inertia, larger gear load, no reach
benefit — usually a pure cost unless extra mass stabilizes terminal precision.
- Higher gear strengthens both motors uniformly: faster approach, but combined with stiff
proportional feedback it invites overshoot and chatter.
- The link lengths also move the elbow and fingertip positions, so the workspace changes
with every length proposal.""",
    "walker2d": """Field physics (thigh_len, leg_len, foot_len = segment half-lengths;
*_density = per-segment material density; gear = motor strength multiplier):
- Longer thighs raise the hip/torso and lengthen the stride, but swing more slowly and put
the torso higher, making balance harder.
- Longer shins lengthen the stance and stride similarly; longer feet change foot placement
relative to the ankle.
- The leg chain moves as a unit: lengthening a thigh automatically pushes the knee and
everything below it farther down, so combined lengths set the total leg length and the
standing height.
- Denser segments add leg/torso inertia and contact impact without adding length.
- Higher gear strengthens all six motors: more authority to recover from imbalance, but
oversized gains then oscillate more easily.""",
    "hopper": """Field physics (thigh_len, leg_len, foot_len = segment half-lengths;
*_density = per-segment material density; gear = motor strength multiplier):
- Longer thigh/leg segments raise the torso and enlarge the hop stride but slow the swing
and raise the center of mass.
- The leg chain moves as a unit: lengthening the thigh pushes the knee and foot downward,
lengthening the shin pushes the foot downward.
- The foot extends forward of the ankle; longer feet give a broader base at touchdown.
- Denser segments add inertia and impact without adding length.
- Higher gear strengthens all three motors: more hop energy per command, but high feedback
gains oscillate more easily.""",
    "half_cheetah": """Field physics (torso_len = torso half-length; bthigh_len, bshin_len,
bfoot_len, fthigh_len, fshin_len, ffoot_len = rear/front segment half-lengths; gear_scale =
multiplier on every motor gear):
- A longer torso moves the rear and front leg attachments apart and shifts the head
forward; leg spacing changes the required front/rear phase relationship.
- Longer rear segments move the rear mass distribution; longer front segments do the same
at the front; asymmetric changes tilt the pitch balance point.
- Segment lengths scale each capsule in place (the cheetah's angled leg layout keeps its
fixed attachment points), so changes are local to the segment.
- gear_scale multiplies all six motors (rear gears are individually stronger than front
gears and stay proportional): raising it adds global authority.""",
    "swimmer": """Field physics (torso_len, mid_len, back_len = segment half-lengths;
radius = body radius; density = material density; gear = motor strength multiplier):
- Longer head pushes the trailing links farther forward with it; longer trailing segments
lengthen the tail. Longer bodies propagate a wave over more length but have more drag and
inertia.
- A wider radius increases volume, mass, and drag together.
- Denser material adds mass and inertia without changing drag.
- Higher gear strengthens both joints: faster wave, but high frequency costs energy in
viscous fluid.""",
    "ant": """Field physics (hip_len = upper-segment reach from torso to hip joint;
ankle_len = lower-segment reach from hip to ankle; density = material density; gear = motor
strength multiplier; torso_radius = torso size; leg_radius = leg thickness):
- Longer upper segments move the hip joints outward; longer lower segments lengthen the
lower legs — together they set standing height, clearance, and stride.
- Denser material raises total mass (torso and all legs) without changing geometry.
- A bigger torso adds body mass; thicker legs add leg mass and contact area.
- Higher gear strengthens all eight motors: more authority over the body, but aggressive
gains can flip the torso.""",
    "swimmer_topology": """Field physics (n_links = number of links in the chain; seg_len =
segment half-length; radius = body radius; density = material density; gear = motor strength
multiplier):
- n_links changes the topology: each extra link appends one trailing segment, one hinge
joint, and one actuator (action dimension grows by one); a longer chain can express a
smoother, larger-amplitude body wave but adds mass, drag, and coordination burden.
- Longer segments lengthen the whole chain; wider radius and higher density add mass and
drag; higher gear strengthens every joint.""",
    "ant_topology": """Field physics (n_legs = number of legs; hip_len = upper-segment reach
to the hip joint; ankle_len = lower-segment reach to the ankle; density = material density;
gear = motor strength multiplier; torso_radius = torso size; leg_radius = leg thickness):
- n_legs changes the topology: legs are spread evenly around the torso; each leg adds two
hinges and two actuators (action dimension grows by two); more legs distribute support and
add weight and coordination burden.
- Longer upper/lower segments raise the body, increase clearance and stride, but raise
inertia and the center of mass; denser material, bigger torso, and thicker legs all add
mass; higher gear strengthens every joint.""",
}

EFFICIENCY_GUIDANCE = """Secondary objectives: minimize squared torque energy
E = sum(dt * ||u||^2) and torque-rate jerk J = sum(dt * ||(u_t - u_{t-1})/dt||^2), and keep
the structure compact (fewer terms). Never sacrifice survival, success, or return for a
small energy or complexity gain; seek Pareto improvements, and when one structure cannot
improve everything, propose distinct variants targeting performance, energy, and
smoothness."""
