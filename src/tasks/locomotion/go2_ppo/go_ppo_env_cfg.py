# Copyright (c) 2026-2027 zh
"""内容：
    创建参考速度跟踪任务基础配置，供机器人环境工厂装配。
"""

import math
from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import GridPatternCfg, ObjRef, RayCastSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.config import ROUGH_TERRAINS_CFG
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

import src.tasks.locomotion.go2_ppo.mdp as mdp

from src.config.go2.go2_config import Go2Cfg

go2_cfg = Go2Cfg()

def make_velocity_env_cfg() -> ManagerBasedRlEnvCfg:
  """创建参考速度跟踪基础任务配置。"""


  # ---------- 传感器 ----------


  terrain_scan = RayCastSensorCfg(
    name = "terrain_scan",
    frame = ObjRef(type="body", name="", entity="robot"),  # 由机器人配置设置。
    ray_alignment = "yaw",
    pattern=GridPatternCfg(size = (1.6, 1.0), resolution = 0.1),
    max_distance = 5.0,
    exclude_parent_body = True,
    debug_vis = True,
    viz = RayCastSensorCfg.VizCfg(show_normals=True),
  )


  # ---------- 观测 ----------


  actor_terms = {
    "base_ang_vel": ObservationTermCfg(
      func = mdp.builtin_sensor,
      params = {"sensor_name": "robot/imu_ang_vel"},
      noise = Unoise(n_min = -0.2, n_max = 0.2),
    ),
    "projected_gravity": ObservationTermCfg(
      func = mdp.projected_gravity,
      noise = Unoise(n_min = -0.05, n_max = 0.05),
    ),
    "command": ObservationTermCfg(
      func = mdp.generated_commands,
      params = {"command_name": "twist"},
    ),
    "phase": ObservationTermCfg(
      func = mdp.phase,
      params = {"period": 0.6, "command_name": "twist"},
    ),
    "joint_pos": ObservationTermCfg(
      func = mdp.joint_pos_rel,
      noise = Unoise(n_min = -0.01, n_max = 0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func = mdp.joint_vel_rel,
      noise = Unoise(n_min = -1.5, n_max = 1.5),
    ),
    "actions": ObservationTermCfg(func = mdp.last_action),
  }

  critic_terms = {
    **actor_terms,
    "base_lin_vel": ObservationTermCfg(
      func = mdp.builtin_sensor,
      params = {"sensor_name": "robot/imu_lin_vel"},
      noise = Unoise(n_min = -0.5, n_max = 0.5),
    ),
    "height_scan": ObservationTermCfg(
      func = envs_mdp.height_scan,
      params = {"sensor_name": "terrain_scan"},
      scale = 1 / terrain_scan.max_distance,
    ),
    "foot_height": ObservationTermCfg(
      func = mdp.foot_height,
      params = {"asset_cfg": SceneEntityCfg("robot", site_names=())},  # 由机器人配置设置。
    ),
    "foot_air_time": ObservationTermCfg(
      func = mdp.foot_air_time,
      params = {"sensor_name": "feet_ground_contact"},
    ),
    "foot_contact": ObservationTermCfg(
      func = mdp.foot_contact,
      params = {"sensor_name": "feet_ground_contact"},
    ),
    "foot_contact_forces": ObservationTermCfg(
      func = mdp.foot_contact_forces,
      params = {"sensor_name": "feet_ground_contact"},
    ),
  }

  observations = {
    "actor": ObservationGroupCfg(
      terms = actor_terms,
      concatenate_terms = True,
      enable_corruption = True,
      history_length = 1,
    ),
    "critic": ObservationGroupCfg(
      terms = critic_terms,
      concatenate_terms = True,
      enable_corruption = False,
      history_length = 1,
    ),
  }


  # ---------- 指标 ----------


  metrics = {
    "mean_action_acc": MetricsTermCfg(
      func = mdp.mean_action_acc,
    ),
  }


  # ---------- 动作 ----------


  base_action_scale = go2_cfg.control.action_scale
  scale = {
    r".*_hip_joint": base_action_scale * go2_cfg.control.hip_reduction,
    r".*_thigh_joint": base_action_scale,
    r".*_calf_joint": base_action_scale,
  }

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name = "robot",
      actuator_names = (".*",),
      scale = scale,
      use_default_offset = True,
    )
  }


  # ---------- 命令 ----------


  commands: dict[str, CommandTermCfg] = {
    "twist": UniformVelocityCommandCfg(
      entity_name = "robot",
      resampling_time_range = (3.0, 8.0),
      rel_standing_envs = 0.05,
      heading_command = True,
      heading_control_stiffness = 0.5,
      debug_vis = True,
      ranges = UniformVelocityCommandCfg.Ranges(
        lin_vel_x = (-1.0, 2.0),
        lin_vel_y = (-1.0, 1.0),
        ang_vel_z = (-1.0, 1.0),
        heading = (-math.pi, math.pi),
      ),
    )
  }


  # ---------- 事件 ----------


  events = {
    # 重置事件。
    "reset_base": EventTermCfg(
      func = mdp.reset_root_state_uniform,
      mode = "reset",
      params = dict(go2_cfg.reset.base_offset)
    ),
    "reset_robot_joints": EventTermCfg(
      func = mdp.reset_joints_by_offset,
      mode = "reset",
      params = {
        "position_range": (-0.0, 0.0),
        "velocity_range": (-0.0, 0.0),
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),

    # 域随机化。
    "push_robot": EventTermCfg(
      func = mdp.push_by_setting_velocity,
      mode = "interval",
      interval_range_s = (5.0, 6.0),
      params = {
        "velocity_range": {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "z": (-0.4, 0.4),
          "roll": (-0.52, 0.52),
          "pitch": (-0.52, 0.52),
          "yaw": (-0.78, 0.78),
        },
      },
    ),
    "foot_friction": EventTermCfg(
      mode = "startup",
      func = dr.geom_friction,
      params = {
        "asset_cfg": SceneEntityCfg("robot", geom_names=()),  # 由机器人配置设置。
        "operation": "abs",
        "ranges": (0.3, 1.6),
        "shared_random": True,  # 所有足端共享同一次采样的摩擦系数。
      },
    ),
    "encoder_bias": EventTermCfg(
      mode = "startup",
      func = dr.encoder_bias,
      params = {
        "asset_cfg": SceneEntityCfg("robot"),
        "bias_range": (-0.015, 0.015),
      },
    ),
    "base_com": EventTermCfg(
      mode = "startup",
      func = dr.body_com_offset,
      params = {
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # 由机器人配置设置。
        "operation": "add",
        "ranges": {
          0: (-0.05, 0.05),
          1: (-0.05, 0.05),
          2: (-0.05, 0.05),
        },
      },
    ),
  }


  # ---------- 奖励 ----------


  rewards = {
    "track_linear_velocity": RewardTermCfg(
      func = mdp.track_linear_velocity,
      weight = 1.0,
      params = {"command_name": "twist", "std": math.sqrt(0.25)},
    ),
    "track_angular_velocity": RewardTermCfg(
      func=mdp.track_angular_velocity,
      weight=1.0,
      params={"command_name": "twist", "std": math.sqrt(0.5)},
    ),
    "body_orientation_l2": RewardTermCfg(
      func=mdp.body_orientation_l2,
      weight=-1.0,
      params={"asset_cfg": SceneEntityCfg("robot", body_names=())},  # 由机器人配置设置。
    ),
    "pose": RewardTermCfg(
      func=mdp.variable_posture,
      weight=1.0,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
        "command_name": "twist",
        "std_standing": {},  # 由机器人配置设置。
        "std_walking": {},  # 由机器人配置设置。
        "std_running": {},  # 由机器人配置设置。
        "walking_threshold": 0.1,
        "running_threshold": 1.5,
      },
    ),
    "body_ang_vel": RewardTermCfg(
      func=mdp.body_angular_velocity_penalty,
      weight=-0.05,  # 可由机器人配置覆盖。
      params={"asset_cfg": SceneEntityCfg("robot", body_names=())},  # 由机器人配置设置。
    ),
    "angular_momentum": RewardTermCfg(
      func=mdp.angular_momentum_penalty,
      weight=-0.025,  # 可由机器人配置覆盖。
      params={"sensor_name": "robot/root_angmom"},
    ),
    "is_terminated": RewardTermCfg(func=mdp.is_terminated, weight=-200.0),
    "joint_acc_l2": RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.5e-7),
    "joint_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-10.0),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.05),
    "foot_gait": RewardTermCfg(
      func=mdp.feet_gait,
      weight=0.5,
      params={
        "period": 0.6,
        "offset": [0.0, 0.5],
        "threshold": 0.56,
        "command_threshold": 0.1,
        "command_name": "twist",
        "sensor_name": "feet_ground_contact",
      }
    ),
    "foot_clearance": RewardTermCfg(
      func=mdp.feet_clearance,
      weight=-1.0,
      params={
        "target_height": 0.10,
        "command_name": "twist",
        "command_threshold": 0.1,
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # 由机器人配置设置。
      },
    ),
    "foot_slip": RewardTermCfg(
      func=mdp.feet_slip,
      weight=-0.25,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "command_threshold": 0.1,
        "asset_cfg": SceneEntityCfg("robot", site_names=()),  # 由机器人配置设置。
      },
    ),
    "soft_landing": RewardTermCfg(
      func=mdp.soft_landing,
      weight=-1e-3,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "command_threshold": 0.1,
      },
    ),
    "stand_still": RewardTermCfg(
      func=mdp.stand_still,
      weight=-1.0,
      params={
        "command_name": "twist",
        "command_threshold": 0.1,
        "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
      },
    ),
  }


  # ---------- 终止 ----------


  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(
      func=mdp.bad_orientation,
      params={"limit_angle": math.radians(70.0)},
    ),
  }


  # ---------- 课程 ----------


  curriculum = {
    "terrain_levels": CurriculumTermCfg(
      func=mdp.terrain_levels_vel,
      params={"command_name": "twist"},
    ),
    "command_vel": CurriculumTermCfg(
      func=mdp.commands_vel,
      params={
        "command_name": "twist",
        "velocity_stages": [
          {"step": 0, "lin_vel_x": (-0.5, 1.0), "lin_vel_y": (-0.5, 0.5), "ang_vel_z": (-1.0, 1.0)},
          {"step": 500 * 24, "lin_vel_x": (-1.0, 1.0), "lin_vel_y": (-1.0, 1.0)},
        ],
      },
    ),
  }


  # ---------- 配置装配 ----------


  return ManagerBasedRlEnvCfg(
    scene = SceneCfg(
      terrain = TerrainEntityCfg(
        terrain_type = "generator",
        terrain_generator = replace(ROUGH_TERRAINS_CFG),
        max_init_terrain_level = 5,
      ),
      sensors = (terrain_scan,),
      num_envs = 1,
      extent = 2.0,
    ),
    observations = observations,
    actions = actions,
    commands = commands,
    events = events,
    rewards = rewards,
    terminations = terminations,
    curriculum = curriculum,
    metrics = metrics,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="",  # 由机器人配置设置。
      distance=3.0,
      elevation=-5.0,
      azimuth=90.0,
    ),
    sim = SimulationCfg(
      nconmax = 35,
      njmax = 1500,
      mujoco = MujocoCfg(
        timestep = 0.005,
        iterations = 10,
        ls_iterations = 20,
      ),
    ),
    decimation = 4,
    episode_length_s = 20.0,
  )
