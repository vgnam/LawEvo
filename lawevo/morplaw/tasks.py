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
    "reacher_payload": """A horizontal two-link planar robot arm must reach random target
points while carrying a concentrated spherical payload at its fingertip. Both revolute joints
are torque actuated, and link lengths, link radii, material densities, motor gear, payload
radius, and payload density are morphology variables. The payload increases distal inertia,
so acceleration and braking authority change much more strongly than when mass is added near
the base. The episode lasts 50 control steps; success requires final Cartesian error below
0.05 m across randomized targets and +/-10% body-mass perturbations.""",
    "reacher_gravity": """A two-link robot arm moves in a vertical x-y plane. Gravity acts
downward along the negative y-axis, so both shoulder and elbow must support configuration-
dependent link weight while reaching random target points. Link lengths, radii, densities,
and motor gear are morphology variables. A controller that works only as a horizontal-plane
kinematic servo can retain steady-state error or sag; proportional task feedback, integral
action, and velocity damping must be balanced. The episode lasts 50 control steps and success
requires final Cartesian error below 0.05 m under +/-10% body-mass perturbations.""",
    "reacher_precision": """A horizontal two-link planar robot arm must acquire a small target
and settle accurately. The target marker radius is 0.003 m and success requires final
fingertip-target error below 0.02 m over a 100-step episode. Link lengths, radii, densities,
and motor gear are morphology variables. High gain can approach quickly but tends to overshoot
or chatter around the tight tolerance, while excessive damping may not settle in time.
Training evaluations perturb body masses by +/-10%.""",
    "pusher": """A fixed-base seven-joint robot arm must contact a movable cylinder on a table
and push it toward a goal. The control signal is a seven-vector of normalized joint commands.
Task-space feedback can separately represent tip-to-object error, object-to-goal error, their
combined push direction, joint damping, and posture regulation. Upper-arm and forearm length
and radius, arm density, and actuator gear are morphology variables, so reach, contact
leverage, inertia, and motor authority must be co-designed. Episodes last 100 control steps,
and success requires the object to finish within 0.1 m of the goal under +/-10% body-mass
perturbations.""",
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
controller law is a shared expression and applies to any number of actuators). Measurable
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
controller law is a shared expression and applies to any gain slots, any leg count), but they add weight
and coordination burden. Gaits coordinate legs in diagonal pairs. Longer legs raise the body
and stride but increase leg inertia and center-of-mass height; stronger gears give more leg
authority. Training evaluations randomly perturb body masses by +/-10%.""",
    "robomorph_flat": """A free-root three-dimensional modular robot must locomote forward
on flat ground. Its body is not a fixed Ant template: a grammar may create a serial chain of
one to four body modules, connect them through rigid, roll, or twist joints, and attach
bilaterally mirrored limbs to any body module. Each limb may contain one to three links with
rigid, roll, knee, or elbow joints and terminate in a contact foot or passive wheel. Every
non-rigid body or limb joint is torque actuated; passive wheels roll freely. The number and
ordering of actuators therefore change with the graph. The controller law is a shared
expression over vector-valued signals, common to every topology. Episodes last 300 control steps
and end early if the root body falls. Training evaluations perturb body masses by +/-10%.""",
    "robomorph_ridged": """A free-root three-dimensional modular robot must locomote forward
across 15 transverse cylindrical ridges. Each ridge has radius 0.2 m and the ridge centers
are spaced 2 m apart. The robot uses the same graph grammar as robomorph_flat: one to four
serial body modules, optional compiler-mirrored bilateral limbs, one to three links per limb,
actuated non-rigid joints, and foot or passive-wheel terminals. The terrain rewards clearance,
well-timed contact, and recovery after impacts; passive wheels must climb repeated rounded
steps rather than merely roll on a plane. Episodes last 300 control steps, end if the root
falls, and training evaluations perturb body masses by +/-10%.""",
    "robomorph_frozen_lake": """A free-root three-dimensional modular robot must locomote
forward on a flat low-friction plane whose tangential friction coefficient is 0.05. The robot
uses the same variable graph grammar as robomorph_flat. Low traction makes narrow, high-force
gaits prone to slip, so body geometry, contact count, passive-wheel use, and the feedback law
must be co-designed for stability and useful thrust. Episodes last 300 control steps, end if
the root falls, and training evaluations perturb body masses by +/-10%.""",
    "robomorph_beams": """A free-root three-dimensional modular robot must locomote forward
through 15 transverse cylindrical beams with radius 0.2 m, center height 0.5 m, and 2 m
spacing. Each beam's lower surface is 0.3 m above the floor, creating repeated overhead
clearance constraints. The robot uses the same variable graph grammar as robomorph_flat.
Compact bodies can pass below the beams while taller bodies may need a crawling posture or
careful contact strategy. Episodes last 300 control steps, end if the root falls, and training
evaluations perturb body masses by +/-10%.""",
    "panda_reach_moving": """A seven-joint Franka Panda arm with a two-finger gripper stands
beside a table and must track a target sphere that never stops moving. The target orbits a
randomized center on a smooth periodic path; the orbit speed is a design parameter of the
environment. The episode lasts 50 control steps. The end-effector command is a bounded
Cartesian displacement. Success requires ending within 0.05 m of the moving goal, and the
dense reward measures instantaneous tracking error, so sustained lag drives the score. A
fast-moving goal punishes purely reactive feedback: the controller must anticipate the
orbit. Training evaluations resample the orbit center and phase every episode.""",
    "panda_push_ice": """A seven-joint Franka Panda arm with a blocked gripper must push a
cube across a table whose lateral friction is a design parameter (default near frictionless).
A fixed cylindrical obstacle stands on the straight path between the cube's start zone and
the goal zone, with a resampled lateral offset every episode. The episode lasts 50 control
steps and the command is a bounded end-effector displacement. Success requires the cube to
finish within 0.05 m of the goal. Low friction means every push keeps sliding: braking,
side-stepping the obstacle, and contact maintenance dominate. Training evaluations resample
the cube, goal, and obstacle placements.""",
    "panda_slide_gate": """A seven-joint Franka Panda arm with a blocked gripper must strike
a low-friction puck through a narrow gate formed by two static walls; the gate opening width
is a design parameter. The goal lies beyond the gate and inside its corridor, out of the
arm's direct reach. The episode lasts 50 control steps with a bounded end-effector
displacement command. Success requires the puck within 0.05 m of the goal. The puck must
pass through the gate first: shots aimed straight at the goal bounce off the walls. The
controller should stage its approach, aim the impulse through the gate center, and avoid
wasting the short horizon on repeated weak strikes. Training evaluations resample the puck,
gate-relative goal, and contact conditions.""",
    "panda_pick_distractor": """A seven-joint Franka Panda arm with a two-finger gripper must
pick up a cube whose mass is a design parameter (heavier than the stock 1.0 kg) and place it
at a randomized goal, while a separate clutter box is resampled near the goal region every
episode. The episode lasts 50 control steps; the command is a bounded end-effector
displacement plus gripper displacement. Success requires the cube within 0.05 m of the goal.
A heavier cube sags during transport, so the grasp must close firmly and lift with extra
clearance, and the transport path must avoid knocking the clutter into the goal. Premature
release under load is the characteristic failure. Training evaluations resample the cube,
goal, and clutter placements.""",
    "panda_stack_narrow": """A seven-joint Franka Panda arm with a two-finger gripper must
stack one cube stably on top of another within 100 control steps. Two design parameters
govern the environment: the success distance tolerance (tighter than the stock task) and
the settle speed below which the stacked cube counts as at rest. A drop that lands inside
tolerance while still moving fails; the controller must lower gently and release only when
the cube is quiet. The command is a bounded end-effector displacement plus gripper
displacement, and the dense reward combines both cube-goal distances. Training evaluations
resample both cube placements.""",
}

CONTROL_GOALS = {
    "reacher": """Primary goal: move the fingertip onto the target quickly and keep it there,
maximizing return over the 50-step episode; final fingertip-target distance below 0.05
counts as success across randomized targets and link masses. Near the target, suppress
overshoot, oscillation, and torque chatter. Among structures with comparable accuracy,
prefer lower torque energy, smoother torque changes, and simpler laws. A fast initial swing
that cannot settle is not success. When designing for THIS body, remember the trade-off the
morphology imposes: long heavy links need strong low-frequency feedback and damping to stop
cleanly, while short light links can use more aggressive proportional action.""",
    "reacher_payload": """Primary goal: move the payload-bearing fingertip onto each target
within 50 steps and finish below 0.05 m error. Match task-space proportional/integral action
and joint damping to the distal payload inertia: enough authority for fast approach, enough
braking to prevent the payload from carrying through the target, and no persistent offset.
Among similarly accurate bodies, prefer lower torque energy, smoother commands, and fewer
terms. Robustness must hold under randomized link and payload mass perturbations.""",
    "reacher_gravity": """Primary goal: reach and hold random targets in the vertical plane
within 50 steps, finishing below 0.05 m error despite gravitational loading. The law must
remove configuration-dependent sag without causing integral windup, overshoot, or torque
chatter. Body geometry and density determine gravity torque, while gear determines available
joint authority. Prefer robust accuracy first, then lower energy, smoother commands, and
lower law/morphology complexity.""",
    "reacher_precision": """Primary goal: enter and remain inside a 0.02 m Cartesian error
ball during the 100-step episode. Optimize settling, not just closest approach: suppress
overshoot, limit cycles, and noisy normalized feedback near zero. Integral action may remove
small offsets but must not wind up; damping must stop the arm without making it sluggish.
Prefer lower energy, jerk, and structural complexity only after tight accuracy is achieved.""",
    "pusher": """Primary goal: first acquire stable fingertip-object contact, then move the
object to within 0.1 m of the goal during the 100-step episode. A useful law must transition
from reaching toward the object to sustained pushing toward the goal without losing contact
or saturating all seven joints. Match arm reach and leverage to the task while damping
redundant joint motion. Prefer task success, then lower energy, smoother commands, and lower
law/morphology complexity.""",
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
or one impulse. Minimize torque energy, jerk, lateral drift, and unnecessary complexity among
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
complexity among comparable swimmers. Because the law is a shared expression that applies to
any actuator count, prefer structures whose signals are naturally periodic and distributed
(sinusoidal phases, posture, damping) over anything that assumes a fixed joint count. For
THIS body, the wave must match the number and length of segments: more links allow a
smoother wave but require the phases to advance evenly along the chain.""",
    "ant_topology": """Primary goal: walk forward quickly, stay healthy for all 300 steps,
and end above 0.75 m/s, for any proposed number of legs. Maintain torso height near 0.65 m
and control roll/pitch with coordinated diagonal-leg cycles. Minimize torque energy and jerk
among equally successful gaits. Because the law is a shared expression that applies to
any actuator count, prefer structures whose signals are naturally periodic and distributed
(sinusoidal phases, posture, damping) over anything that assumes a fixed leg count. For THIS
body, gaits must respect the leg count and geometry: with more legs, diagonal pairing and
phase offsets must scale accordingly.""",
    "robomorph_flat": """Primary goal: discover a high-performing body plan and matched
symbolic law that move forward for all 300 steps without the root body falling. Structural
changes may be non-local: body count, body joints, limb placement, limb depth, joint types,
and foot versus passive wheel are all design variables. Prefer forward return and survival,
then lower mass-normalized torque energy, smoother commands, and lower graph/control
complexity. The law must remain meaningful for the candidate's live actuator graph rather
than assuming a fixed Ant joint order.""",
    "robomorph_ridged": """Primary goal: discover a body plan and matched symbolic law that
make sustained forward progress over repeated 0.2 m-radius ridges and remain healthy for all
300 steps. Favor sufficient underbody clearance, stable contacts before and after each ridge,
and a gait that does not spend all its energy striking obstacles. Evaluate wheels as climbing
contacts, not as an assumed flat-ground advantage. Among similarly capable designs, prefer
lower mass-normalized torque energy, smoother commands, and lower graph/control complexity.""",
    "robomorph_frozen_lake": """Primary goal: discover a body plan and matched symbolic law
that make sustained forward progress for 300 steps despite a 0.05 tangential-friction floor.
Favor slip-tolerant stability, controlled force application, and contact geometry that creates
useful propulsion without relying on high traction. Reject laws that simply saturate motors
and spin or skate in place. Among similarly capable designs, prefer lower mass-normalized
torque energy, smoother commands, and lower graph/control complexity.""",
    "robomorph_beams": """Primary goal: discover a body plan and matched symbolic law that
advance beneath or negotiate repeated beams whose lower surfaces are 0.3 m above the floor,
while remaining healthy for all 300 steps. Favor a compatible resting height, compact pitch
envelope, and a gait that preserves clearance as it moves. Reject designs that gain speed on
open ground but repeatedly strike or become trapped by beams. Among similarly capable designs,
prefer lower mass-normalized torque energy, smoother commands, and lower complexity.""",
    "panda_reach_moving": """Primary goal: track the orbiting goal with minimal sustained
error across all 50 steps and finish inside the 0.05 m tolerance. The goal_velocity signal
measures the orbit directly, so a K-weighted feedforward term can cancel the phase lag that
pure proportional tracking accumulates; the phase signals offer a second, model-free route.
For THIS environment, a faster goal orbit (larger goal_speed) widens the lag that reactive
laws suffer and shifts the payoff toward anticipation; slower orbits reward gentle, smooth
servos. Prefer lower command energy, smoother commands, and fewer signals once accuracy
holds.""",
    "panda_push_ice": """Primary goal: bring the cube inside the 0.05 m goal tolerance within
50 steps while never hitting the obstacle. For THIS environment, a lower table_friction makes
every push glide farther, so braking through damping and early deceleration dominate
acceleration; a higher friction approaches the stock pushing task where contact staging
matters more. The obstacle_repel signal should bend the push path around the cylinder.
Success and dense return dominate; then prefer low command energy, smooth commands,
obstacle-free paths, and compact laws.""",
    "panda_slide_gate": """Primary goal: slide the puck through the gate and inside the
0.05 m goal tolerance within 50 steps. For THIS environment, a narrower gate (smaller
gate_width) demands a precisely aimed impulse and punishes wall bounces; a wider gate
forgives sloppy heading but still requires staging through the opening. Use the
through_gate signal to aim at the gate first and the goal after crossing. Among comparable
shots prefer lower command energy, fewer impacts, a smoother approach, and simpler laws.""",
    "panda_pick_distractor": """Primary goal: place the cube inside the 0.05 m goal tolerance
within 50 steps while keeping the clutter box clear of the goal. For THIS environment, a
heavier cube (larger cube_mass) sags more during transport, demanding a firmer grasp, higher
lift clearance, and gentler horizontal acceleration; a lighter cube approaches the stock
pick-and-place. Use distractor_error to steer the transport path around the clutter. Success
and dense return dominate; then minimize command energy, jerk, failed grasps, premature
releases under load, and unnecessary terms.""",
    "panda_stack_narrow": """Primary goal: stack the cube satisfying both the tightened
distance tolerance and the at-rest settle requirement within 100 steps. For THIS
environment, a smaller distance_threshold or a smaller settle_speed tightens the placement
window: the controller must lower gently through the final centimeters and gate the release
on the settle_velocity signal. A drop that lands inside tolerance while still moving is not
a success. Success and dense return dominate; then minimize command energy, jerk,
collisions with the fixed cube, regrasping, and structural complexity.""",
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
    "pusher": """All signals are seven-vectors aligned with the arm's actuated joints.
- jt_object_error: J^T(object - fingertip), task-space feedback that acquires contact.
- jt_goal_error: J^T(goal - object), projected goal direction for the current arm pose.
- jt_push_error: J^T[(object - fingertip) + 0.5(goal - object)], balancing acquisition and push.
- joint_velocity: joint-rate damping across all seven joints.
- integral_jt_push: clipped integral of jt_push_error; removes persistent push offset but can
  wind up when contact or reach constraints prevent motion.
- tanh_jt_push / normalized_jt_push: bounded and unit-direction versions of push feedback.
- task_damping: J^T J qdot, damping motion of the end effector rather than every null-space
  joint equally.
- posture_error: negative joint positions, a regularizer for redundant arm configurations.""",
    "locomotion": """All signals are vectors with one component per actuator.
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
    "swimmer": """All signals are vectors with one component per actuator.
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
    "ant": """All signals are vectors with one component per actuator (hip then ankle of each
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
    "robomorph": """All signals are vectors aligned with the live MuJoCo actuator order; no
signal assumes a fixed number of legs or a fixed joint naming layout.
- phase_sin / phase_cos: a fixed-frequency CPG. Mirrored limb joints receive side- and
  body-position-aware phase offsets; actuated body joints alternate by graph order.
- posture_error / joint_velocity: position regulation and damping read through MuJoCo's
  actuator-to-joint map, so passive wheel coordinates are intentionally excluded.
- integral_posture: clipped time integral of actuated-joint posture error.
- tanh_posture / tanh_velocity: bounded posture and damping feedback.
- body_angle: root roll/pitch projected onto side/front correction patterns for each motor.
- height_error: root-height deficit broadcast to every actuator.
- forward_speed_error: target-speed deficit with graph-order gait signs.""",
    "panda_reach_moving": """All signals are three-vectors broadcast over the Cartesian
end-effector action, except the phase pair which is scalar-like.
- goal_error: instantaneous end-effector-to-goal offset — the proportional tracking term.
- normalized_goal_error: unit-direction version; useful for constant-speed pursuit.
- integral_goal_error: clipped error integral — removes steady lag but winds up on a moving
  goal if unsaturated.
- goal_velocity: measured goal motion per control step — the feedforward term that reactive
  laws lack; on a periodic orbit it is itself nearly periodic.
- phase_sin / phase_cos: elapsed-time oscillators — a model-free anticipation channel for the
  periodic orbit.
- eef_damping: negative end-effector velocity — smooths the pursuit and prevents chatter.
- tanh_goal_error: saturated error — strong bounded feedback near the goal.""",
    "panda_push_ice": """All signals are three-vectors aligned with the Cartesian action.
- reach_object: cube-to-end-effector offset — acquires contact.
- normalized_reach_object: unit-direction reach — constant-magnitude approach.
- object_goal_error: goal-to-cube offset — the push direction once contact is established.
- normalized_object_goal_error: unit push direction.
- contact_then_goal: reaches the cube until it is near, then switches to the push direction
  — the staged sequential term.
- obstacle_repel: unit vector from the obstacle toward the cube — bends the push path around
  the cylinder; on a low-friction table even a slight heading error becomes a large miss.
- eef_damping: negative end-effector velocity — the only brake the controller has, since the
  table friction is near zero.""",
    "panda_slide_gate": """All signals are three-vectors aligned with the Cartesian action.
- reach_object: puck-to-end-effector offset — positions the strike.
- object_goal_error: goal-to-puck offset — the naive shot direction that ignores the gate.
- contact_then_goal: staged approach toward the puck, then the goal.
- through_gate: points from the puck to the gate center before the puck crosses the gate
  line and to the goal after — the staging signal that separates gate-aware laws from
  wall-bouncing ones.
- eef_damping: negative end-effector velocity — stabilizes the wind-up before the impulse.""",
    "panda_pick_distractor": """All signals are four-vectors (Cartesian xyz plus gripper).
- reach_object: cube-to-end-effector offset plus neutral gripper — approach.
- object_goal_error: goal-to-cube offset — the transport direction.
- eef_damping: negative end-effector velocity — damps the sag-induced sway of a heavy cube.
- grasp_close: closes the gripper; a heavier cube (cube_mass) needs a firm, early closure.
- lift_then_transport: raises above the sag height before horizontal motion.
- release_on_target: opens the gripper only near the goal.
- pick_place_sequence: the full staged sequence.
- distractor_error: end-effector-to-clutter offset — lets the law steer the transport path
  away from the clutter box instead of knocking it into the goal.""",
    "panda_stack_narrow": """All signals are four-vectors (Cartesian xyz plus gripper).
- reach_cube_one: movable-cube-to-end-effector offset — approach.
- cube_one_goal_error: stack-goal-to-cube offset — the placement direction.
- eef_damping: negative end-effector velocity — smooths the final descent.
- grasp_close: closes the gripper around the movable cube.
- lift_then_stack: raises to the stack clearance height before horizontal alignment.
- release_on_stack: opens the gripper once aligned above the fixed cube.
- stack_sequence: the staged reach-grasp-lift-align-lower-release sequence.
- settle_velocity: negative stacked-cube velocity — gates the release on the at-rest
  requirement; with a tight distance_threshold, only gentle placement survives.""",
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
    "reacher_payload": """Field physics extends the base reacher fields with payload_radius
and payload_density:
- Endpoint payload mass scales with density and approximately with radius cubed, so a small
  radius increase can sharply increase distal inertia and stopping distance.
- Longer links compound payload inertia through a larger moment arm; increasing link length
  and payload size together demands much more motor and damping authority.
- Higher gear can restore acceleration but may amplify overshoot and chatter unless the law
  adds matched velocity damping.
- Link radii/densities distribute mass along the arm, whereas payload mass is concentrated at
  the fingertip and therefore has stronger dynamic leverage.""",
    "reacher_gravity": """Field physics in the vertical plane (l0/l1 lengths, r0/r1 radii,
density0/density1 material density, gear motor strength):
- Longer and denser links raise both rotational inertia and configuration-dependent gravity
  torque; the distal link also loads the shoulder through the full kinematic chain.
- Increasing gear improves gravity compensation authority, but high proportional/integral
  gains can overshoot as the arm moves through configurations with different gravity load.
- Thicker links add mass roughly with radius squared, increasing sag without increasing reach.
- Morphology and integral action interact strongly: underpowered heavy arms need sustained
  bias, but excessive integral feedback winds up near unreachable or saturated states.""",
    "reacher_precision": """Field physics for tight settling uses the base reacher fields:
- Short/light links have lower inertia and can settle quickly, but too little combined length
  makes outer targets unreachable.
- Longer or denser links require more braking and make a 0.02 m tolerance harder to hold.
- Higher gear shortens rise time but magnifies quantization-free numerical chatter and
  overshoot when paired with aggressive task-space gains.
- Geometry changes the Jacobian and conditioning near each target, so one PID-like law must
  remain well behaved across the evolved arm's workspace.""",
    "pusher": """Field physics (upper_len/forearm_len segment lengths, upper_radius/
forearm_radius thickness, arm_density material density, gear motor-strength scale):
- Longer links expand the reachable table area and can improve pushing leverage, but increase
  inertia and can make near-object positioning less precise.
- upper_len moves the elbow and forearm_len moves the wrist/end effector, so both are coupled
  kinematic changes rather than visual scaling.
- Thicker or denser arms resist contact disturbances but cost more torque and are harder to
  stop; distal forearm mass has especially strong leverage.
- Higher gear helps maintain object contact and push force, but can cause impact, slip, and
  joint chatter when task-space gains are too aggressive.
- All seven joints remain actuated; morphology changes geometry and dynamics, not the action
  dimension.""",
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
    "robomorph_flat": """Grammar physics:
- Adding body modules increases length and offers more limb attachment sites, but adds mass
  and may require actuated roll/twist stabilization between modules.
- Bilateral limbs are mirrored automatically. More limb pairs enlarge the support polygon
  but add mass and coordination burden; limb placement along the body controls pitch balance.
- Longer/deeper limbs increase clearance and stride but raise swing inertia and root height.
- Knee joints swing in the sagittal plane, roll joints provide lateral articulation, elbow
  joints turn in the horizontal plane, and rigid joints add structure without an actuator.
- Passive wheels can exploit rolling contact on flat ground but cannot generate torque; an
  actuated upstream joint must load and steer them.
- A graph must contain 2-16 actuated joints. Compile-time bilateral symmetry and bounded
  module counts keep generated designs physically valid.""",
    "robomorph_ridged": """Grammar physics on repeated ridges:
- The same body/limb grammar and 2-16 actuator constraint as robomorph_flat apply.
- Ridges have 0.2 m radius and 2 m center spacing. Limb depth and body height determine
  whether the torso clears each crest; longer limbs add clearance but also swing inertia.
- Multiple limb pairs can bridge a ridge and stabilize impacts, but add mass and coordination
  burden. Their body-chain attachment sites determine pitch leverage during climbing.
- Passive wheels reduce rolling loss between ridges but still require upstream actuation and
  enough radius, loading, and approach geometry to climb a 0.2 m rounded obstacle.
- Articulated body joints can conform to crests or recover pitch, but consume actuator budget
  and can buckle without sufficient posture feedback.""",
    "robomorph_frozen_lake": """Grammar physics on low-friction ground:
- The same body/limb grammar and 2-16 actuator constraint as robomorph_flat apply.
- Floor tangential friction is 0.05, so aggressive fore-aft contact forces readily become
  slip. Wider bilateral support, lower root height, and distributed contacts improve stability.
- Extra limb pairs may share contact load but add inertial and coordination costs. Long limbs
  raise the center of mass and amplify lateral slip unless the law damps roll and pitch.
- Passive wheels are unpowered and low traction does not guarantee useful rolling propulsion;
  upstream joints must generate a controlled normal load and favorable contact direction.
- Prefer morphology-law pairs that apply force smoothly instead of depending on impulsive
  push-off or motor saturation.""",
    "robomorph_beams": """Grammar physics under repeated beams:
- The same body/limb grammar and 2-16 actuator constraint as robomorph_flat apply.
- Beam centers are 0.5 m high with radius 0.2 m, leaving only 0.3 m beneath each beam. Root
  height, body thickness, limb depth, and gait-induced pitching jointly determine clearance.
- Shorter or laterally spread limbs and a compact body chain can lower the profile; longer
  limbs may improve stride but raise the torso into the obstacle corridor.
- Additional body articulation can create a crawling posture, but roll/twist joints add
  instability and require matched posture feedback.
- Passive wheels may help a low robot roll between beams, but an upstream joint must load and
  steer them and the whole body must remain below the clearance envelope.""",
    "panda_reach_moving": """Field physics (goal_speed = tangential speed of the orbiting
goal in m/s):
- The parameter does not change the robot; it changes the world the controller faces. A
  faster goal stretches the tracking lag that any reactive law accumulates, shifting the
  payoff toward feedforward and periodic anticipation terms.
- Slower goals flatter simple proportional servos but leave less return on the table for
  laws that can anticipate.
- The arm, table, and orbit amplitude are fixed; only the demand on the controller moves.""",
    "panda_push_ice": """Field physics (table_friction = lateral friction coefficient of the
table, cube, and obstacle):
- Lower friction makes the cube glide after every push: identical commands travel farther
  and stop later, so braking authority (damping feedback) and early deceleration matter more
  than pushing force.
- Very low friction rewards gentle, repeated corrections; higher friction approaches the
  stock pushing task where maintaining contact through the push dominates.
- The obstacle sits between the start and goal zones with a resampled lateral offset, so
  laws that never bend the push path collide regardless of friction.""",
    "panda_slide_gate": """Field physics (gate_width = opening between the two static walls
in meters):
- A narrower gate shrinks the corridor the puck must pass through, so the impulse heading
  must be aimed precisely; wall bounces become fatal to the episode budget.
- A wider gate forgives heading errors but the through-gate staging signal still separates
  gate-aware laws from straight-at-goal shots that strike the wall face.
- The puck's low friction and the goal's placement inside the gate corridor are fixed; only
  the corridor width moves.""",
    "panda_pick_distractor": """Field physics (cube_mass = mass of the grasped cube in kg):
- A heavier cube sags more under the gripper during transport: the arm droops, horizontal
  acceleration must fall, and the grasp must close earlier and firmer or the cube slips.
- Near the stock 1.0 kg the task approaches the standard pick-and-place; toward 3.0 kg the
  lift clearance and gentle-transport structure of the law carry the episode.
- The clutter box is resampled near the goal every episode; its positions are not a design
  variable, so the law must handle the distraction, not the environment.""",
    "panda_stack_narrow": """Field physics (distance_threshold = success distance tolerance
in meters; settle_speed = maximum stacked-cube speed in m/s for the placement to count):
- Both parameters tighten the placement window together: a smaller tolerance demands precise
  horizontal alignment, while a smaller settle_speed demands a gentle, damped descent.
- Loose settings approach the stock stack task; tight settings make dropping-from-height
  useless because the bounce violates the settle requirement.
- The release timing — ideally gated on the settle_velocity signal — is where a law wins or
  loses under tight settings.""",
}

EFFICIENCY_GUIDANCE = """Secondary objectives: minimize squared torque energy
E = sum(dt * ||u||^2) and torque-rate jerk J = sum(dt * ||(u_t - u_{t-1})/dt||^2), and keep
the structure compact (fewer structural nodes). Never sacrifice survival, success, or return for a
small energy or complexity gain; seek Pareto improvements, and when one structure cannot
improve everything, propose distinct variants targeting performance, energy, and
smoothness."""
