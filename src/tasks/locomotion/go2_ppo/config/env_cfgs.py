# Copyright (c) 2026-2027 zh
"""
内容：
    使用同一份 Go2 参数装配粗糙地形、平地及播放模式环境。
"""

from src.assets.robots import (
  get_go2_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers import TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from src.config.go2.go2_config import Go2Cfg
from src.tasks.locomotion.go2_ppo.go_ppo_env_cfg import make_velocity_env_cfg


def unitree_go2_rough_env_cfg(
  play: bool = False,
  *,
  go2_cfg: Go2Cfg | None = None,
) -> ManagerBasedRlEnvCfg:
  """创建 Unitree Go2 粗糙地形速度配置。"""
  go2_cfg = go2_cfg if go2_cfg is not None else Go2Cfg()
  cfg = make_velocity_env_cfg(go2_cfg)

  cfg.scene.entities = {"robot": get_go2_robot_cfg(go2_cfg)}

  # 将射线传感器坐标系设置为 Go2 的 base_link。
  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.frame.name = "base_link"

  foot_names = ("FR", "FL", "RR", "RL")
  site_names = ("FR", "FL", "RR", "RL")
  geom_names = tuple(f"{name}_foot_collision" for name in foot_names)

  feet_ground_cfg = ContactSensorCfg(
    name = "feet_ground_contact",
    primary = ContactMatch(mode = "geom", pattern = geom_names, entity = "robot"),
    secondary = ContactMatch(mode = "body", pattern = "terrain"),
    fields = ("found", "force"),
    reduce = "netforce",
    num_slots = go2_cfg.sensors.contact_num_slots,
    track_air_time=True,
  )

  nonfoot_ground_cfg = ContactSensorCfg(
    name = "nonfoot_ground_touch",
    primary = ContactMatch(
      mode = "geom",
      entity = "robot",
      # 获取所有碰撞几何体。
      pattern = r".*_collision\d*$",
      # 排除足端几何体。
      exclude = tuple(geom_names),
    ),
    secondary = ContactMatch(mode = "body", pattern = "terrain"),
    fields = ("found", "force"),
    reduce = "none",
    num_slots = go2_cfg.sensors.contact_num_slots,
    history_length = go2_cfg.sensors.nonfoot_history_length,
  )

  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    nonfoot_ground_cfg,
  )

  cfg.viewer.body_name = "base_link"

  cfg.observations["critic"].terms["foot_height"].params["asset_cfg"].site_names = site_names

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  cfg.events["foot_contact_softness"].params["asset_cfg"].geom_names = geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)
  cfg.events["link_mass"].params["asset_cfg"].body_names = (r"[FR][LR]_(hip|thigh|calf)",)
  cfg.events["external_force"].params["asset_cfg"].body_names = ("base_link",)

  cfg.rewards["body_orientation_l2"].params["asset_cfg"].body_names = ("base_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("base_link",)
  cfg.rewards["foot_clearance"].params["asset_cfg"].site_names = site_names
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = site_names

  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={
      "sensor_name": nonfoot_ground_cfg.name,
      "force_threshold": go2_cfg.reward.illegal_contact_force_threshold,
    },
  )

  # 应用播放模式覆盖配置。
  if play:
    # 将回合长度设置为近似无限。
    cfg.episode_length_s = go2_cfg.play.episode_length_s

    cfg.observations["actor"].enable_corruption = False
    # 播放使用标称动力学及传感器标定，避免训练扰动影响策略观察。
    for name in (
      "push_robot", "foot_friction", "encoder_bias", "base_com", "link_mass",
      "pd_gains", "effort_limits", "joint_friction", "joint_armature",
      "joint_damping", "imu_errors", "external_force", "foot_contact_softness",
    ):
      cfg.events.pop(name, None)
    cfg.curriculum = {}
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = tuple(go2_cfg.commands.ranges.lin_vel_x)
    twist_cmd.ranges.lin_vel_y = tuple(go2_cfg.commands.ranges.lin_vel_y)
    twist_cmd.ranges.ang_vel_z = tuple(go2_cfg.commands.ranges.ang_vel_yaw)
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = go2_cfg.play.terrain_num_cols
        cfg.scene.terrain.terrain_generator.num_rows = go2_cfg.play.terrain_num_rows
        cfg.scene.terrain.terrain_generator.border_width = go2_cfg.play.terrain_border_width

  return cfg


def unitree_go2_flat_env_cfg(
  play: bool = False,
  *,
  go2_cfg: Go2Cfg | None = None,
) -> ManagerBasedRlEnvCfg:
  """创建 Unitree Go2 平地速度配置。"""
  go2_cfg = go2_cfg if go2_cfg is not None else Go2Cfg()
  cfg = unitree_go2_rough_env_cfg(play=play, go2_cfg=go2_cfg)

  cfg.sim.njmax = go2_cfg.sim.flat.njmax
  cfg.sim.mujoco.ccd_iterations = go2_cfg.sim.flat.ccd_iterations
  cfg.sim.contact_sensor_maxmatch = go2_cfg.sim.flat.contact_sensor_maxmatch
  cfg.sim.nconmax = go2_cfg.sim.flat.nconmax

  # 切换到平地。
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  # 平地无需地形扫描，因此移除射线传感器和高度扫描观测。
  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )
  cfg.observations["critic"].terms.pop("height_scan", None)

  # 关闭地形课程；播放模式下粗糙地形配置已清空课程。
  cfg.curriculum.pop("terrain_levels", None)

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = tuple(go2_cfg.play.lin_vel_x)
    twist_cmd.ranges.lin_vel_y = tuple(go2_cfg.play.lin_vel_y)
    twist_cmd.ranges.ang_vel_z = tuple(go2_cfg.play.ang_vel_yaw)

  return cfg
