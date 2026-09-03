from __future__ import annotations

import math

from lawevo.morplaw.morphology import (
    KIND_COUNT,
    KIND_GEAR,
    KIND_LENGTH,
    KIND_MASS,
    KIND_RADIUS,
    MorphologyField,
    MorphologyTemplate,
    _format_number,
)


class SwimmerTopologyTemplate(MorphologyTemplate):
    """Topology-changing Swimmer: n_links body segments (3..6).

    Each extra link appends one hinge joint and one actuator, so the observation
    and action dimensions grow with the topology. The law space is unchanged:
    laws are free-form expressions over vector-valued signals, so CEM adapts
    its gain slots to the new dimension.
    """

    env_id = "Swimmer-v5"
    asset_path = None
    fields = (
        MorphologyField(
            "n_links",
            KIND_COUNT,
            3.0,
            (3.0, 6.0),
            "body links; each extra link adds one hinge joint and one actuator",
        ),
        MorphologyField("seg_len", KIND_LENGTH, 0.5, (0.3, 0.7), "m segment halflength"),
        MorphologyField("radius", KIND_RADIUS, 0.1, (0.06, 0.14), "m"),
        MorphologyField("density", KIND_MASS, 1000.0, (500.0, 2000.0), "kg/m^3"),
        MorphologyField("gear", KIND_GEAR, 150.0, (75.0, 300.0), "actuator gear"),
    )

    def _xml(self, values: dict[str, float]) -> str:
        fmt = _format_number
        n_links = int(values["n_links"])
        seg = values["seg_len"]
        seg_length = 2.0 * seg
        radius = fmt(values["radius"])
        density = fmt(values["density"])
        gear = fmt(values["gear"])
        # Build the chain from the innermost segment outward: each body contains the next.
        inner = ""
        for index in range(n_links - 1, 0, -1):
            name = {1: "mid", 2: "back"}.get(index, f"seg_{index}")
            pos_x = seg if index == 1 else -seg_length
            inner = (
                f'      <body name="{name}" pos="{fmt(pos_x)} 0 0">\n'
                f'        <geom density="{density}" fromto="0 0 0 {fmt(-seg_length)} 0 0" '
                f'size="{radius}" type="capsule"/>\n'
                f'        <joint axis="0 0 1" limited="true" name="motor{index}_rot" '
                f'pos="0 0 0" range="-100 100" type="hinge"/>\n'
                + inner
                + "      </body>\n"
            )
        actuators = "\n".join(
            f'    <motor ctrllimited="true" ctrlrange="-1 1" gear="{gear}" joint="motor{index}_rot"/>'
            for index in range(1, n_links)
        )
        return f"""<mujoco model="swimmer">
  <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>
  <option density="4000" integrator="RK4" timestep="0.01" viscosity="0.1"/>
  <default>
    <geom conaffinity="0" condim="1" contype="0" material="geom" rgba="0.8 0.6 .4 1"/>
    <joint armature="0.1"/>
  </default>
  <asset>
    <texture builtin="gradient" height="100" rgb1="1 1 1" rgb2="0 0 0" type="skybox" width="100"/>
    <texture builtin="flat" height="1278" mark="cross" markrgb="1 1 1" name="texgeom" random="0.01" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" type="cube" width="127"/>
    <texture builtin="checker" height="100" name="texplane" rgb1="0 0 0" rgb2="0.8 0.8 0.8" type="2d" width="100"/>
    <material name="MatPlane" reflectance="0.5" shininess="1" specular="1" texrepeat="30 30" texture="texplane"/>
    <material name="geom" texture="texgeom" texuniform="true"/>
  </asset>
  <worldbody>
    <light cutoff="100" diffuse="1 1 1" dir="-0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3" specular=".1 .1 .1"/>
    <geom condim="3" material="MatPlane" name="floor" pos="0 0 -0.1" rgba="0.8 0.9 0.8 1" size="40 40 0.1" type="plane"/>
    <body name="torso" pos="0 0 0">
      <camera name="track" mode="trackcom" pos="0 -3 3" xyaxes="1 0 0 0 1 1"/>
      <geom density="{density}" fromto="{fmt(seg + 1.0)} 0 0 {fmt(seg)} 0 0" size="{radius}" type="capsule"/>
      <joint axis="1 0 0" name="slider1" pos="0 0 0" type="slide"/>
      <joint axis="0 1 0" name="slider2" pos="0 0 0" type="slide"/>
      <joint axis="0 0 1" name="free_body_rot" pos="0 0 0" type="hinge"/>
{inner}    </body>
  </worldbody>
  <actuator>
{actuators}
  </actuator>
</mujoco>
"""


class AntTopologyTemplate(MorphologyTemplate):
    """Topology-changing Ant: n_legs legs (4..6), evenly spread around the torso.

    Each leg adds two hinge joints (hip, ankle) and two actuators. The adapter
    derives its per-actuator patterns from the resulting action dimension.
    """

    env_id = "Ant-v5"
    asset_path = None
    fields = (
        MorphologyField(
            "n_legs",
            KIND_COUNT,
            4.0,
            (4.0, 6.0),
            "legs; each leg adds two hinge joints and two actuators",
        ),
        MorphologyField("hip_len", KIND_LENGTH, 0.2, (0.1, 0.3), "m hip reach scale"),
        MorphologyField("ankle_len", KIND_LENGTH, 0.4, (0.2, 0.6), "m ankle reach scale"),
        MorphologyField("density", KIND_MASS, 5.0, (2.5, 10.0), "kg/m^3"),
        MorphologyField("gear", KIND_GEAR, 150.0, (75.0, 300.0), "actuator gear"),
        MorphologyField("torso_radius", KIND_RADIUS, 0.25, (0.15, 0.35), "m"),
        MorphologyField("leg_radius", KIND_RADIUS, 0.08, (0.05, 0.12), "m"),
    )

    def _xml(self, values: dict[str, float]) -> str:
        fmt = _format_number
        n_legs = int(values["n_legs"])
        # Default 4-leg reach is (0.2, 0.2) per axis, i.e. magnitude 0.2 * sqrt(2).
        hip_reach = values["hip_len"] * math.sqrt(2.0)
        ankle_reach = values["ankle_len"] * math.sqrt(2.0)
        leg_radius = fmt(values["leg_radius"])
        legs = []
        actuators = []
        init_joints = []
        for index in range(n_legs):
            leg_number = index + 1
            angle = math.radians(45.0 + index * 360.0 / n_legs)
            direction_x, direction_y = math.cos(angle), math.sin(angle)
            hip_x, hip_y = hip_reach * direction_x, hip_reach * direction_y
            ankle_x, ankle_y = ankle_reach * direction_x, ankle_reach * direction_y
            axis_x = -1 if leg_number % 2 == 1 else 1
            ankle_range = "30 70" if (leg_number % 4) <= 1 else "-70 -30"
            ankle_init = 1.0 if (leg_number % 4) <= 1 else -1.0
            legs.append(
                f'      <body name="leg_{leg_number}" pos="0 0 0">\n'
                f'        <geom fromto="0 0 0 {fmt(hip_x)} {fmt(hip_y)} 0" '
                f'name="aux_{leg_number}_geom" size="{leg_radius}" type="capsule"/>\n'
                f'        <body name="aux_{leg_number}" pos="{fmt(hip_x)} {fmt(hip_y)} 0">\n'
                f'          <joint axis="0 0 1" name="hip_{leg_number}" pos="0 0 0" '
                f'range="-30 30" type="hinge"/>\n'
                f'          <geom fromto="0 0 0 {fmt(hip_x)} {fmt(hip_y)} 0" '
                f'name="leg_{leg_number}_geom" size="{leg_radius}" type="capsule"/>\n'
                f'          <body pos="{fmt(hip_x)} {fmt(hip_y)} 0">\n'
                f'            <joint axis="{axis_x} 1 0" name="ankle_{leg_number}" pos="0 0 0" '
                f'range="{ankle_range}" type="hinge"/>\n'
                f'            <geom fromto="0 0 0 {fmt(ankle_x)} {fmt(ankle_y)} 0" '
                f'name="ankle_{leg_number}_geom" size="{leg_radius}" type="capsule"/>\n'
                f'          </body>\n'
                f'        </body>\n'
                f'      </body>\n'
            )
            actuators.append(
                f'<motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="hip_{leg_number}" '
                f'gear="{fmt(values["gear"])}"/>'
            )
            actuators.append(
                f'<motor ctrllimited="true" ctrlrange="-1.0 1.0" joint="ankle_{leg_number}" '
                f'gear="{fmt(values["gear"])}"/>'
            )
            init_joints.append(f"0.0 {fmt(ankle_init)}")
        return f"""<mujoco model="ant">
  <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>
  <option integrator="RK4" timestep="0.01"/>
  <custom>
    <numeric data="0.0 0.0 0.55 1.0 0.0 0.0 0.0 {' '.join(init_joints)}" name="init_qpos"/>
  </custom>
  <default>
    <joint armature="1" damping="1" limited="true"/>
    <geom conaffinity="0" condim="3" density="{fmt(values['density'])}" friction="1 0.5 0.5" margin="0.01" rgba="0.8 0.6 0.4 1"/>
  </default>
  <asset>
    <texture builtin="gradient" height="100" rgb1="1 1 1" rgb2="0 0 0" type="skybox" width="100"/>
    <texture builtin="flat" height="1278" mark="cross" markrgb="1 1 1" name="texgeom" random="0.01" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" type="cube" width="127"/>
    <texture builtin="checker" height="100" name="texplane" rgb1="0 0 0" rgb2="0.8 0.8 0.8" type="2d" width="100"/>
    <material name="MatPlane" reflectance="0.5" shininess="1" specular="1" texrepeat="60 60" texture="texplane"/>
    <material name="geom" texture="texgeom" texuniform="true"/>
  </asset>
  <worldbody>
    <light cutoff="100" diffuse="1 1 1" dir="-0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3" specular=".1 .1 .1"/>
    <geom conaffinity="1" condim="3" material="MatPlane" name="floor" pos="0 0 0" rgba="0.8 0.9 0.8 1" size="40 40 40" type="plane"/>
    <body name="torso" pos="0 0 0.75">
      <camera name="track" mode="trackcom" pos="0 -3 0.3" xyaxes="1 0 0 0 0 1"/>
      <geom name="torso_geom" pos="0 0 0" size="{fmt(values['torso_radius'])}" type="sphere"/>
      <joint armature="0" damping="0" limited="false" margin="0.01" name="root" pos="0 0 0" type="free"/>
{''.join(legs)}    </body>
  </worldbody>
  <actuator>
{chr(10).join(actuators)}
  </actuator>
</mujoco>
"""
