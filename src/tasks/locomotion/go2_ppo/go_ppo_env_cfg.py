# Copyright (c) 2026-2027 zh
"""
内容：
    将集中定义的 Go2 参数装配为基础速度跟踪任务配置。
"""

from copy import deepcopy
import math

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
from mjlab.terrains import TerrainEntityCfg, TerrainGeneratorCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

import src.tasks.locomotion.go2_ppo.mdp as mdp
from src.tasks.locomotion.go2_ppo.mdp.velocity_command import UniformVelocityCommandCfg

from src.config.go2.go2_config import Go2Cfg


def make_velocity_env_cfg(go2_cfg: Go2Cfg | None = None) -> ManagerBasedRlEnvCfg:
  """创建基础速度跟踪任务配置。"""

  go2_cfg = go2_cfg if go2_cfg is not None else Go2Cfg()
  env_cfg = go2_cfg.env
  sim_cfg = go2_cfg.sim
  sensor_cfg = go2_cfg.sensors
  terrain_cfg = go2_cfg.terrain
  viewer_cfg = go2_cfg.viewer
  control_cfg = go2_cfg.control
  command_cfg = go2_cfg.commands
  domain_rand_cfg = go2_cfg.domain_rand
  mdp.validate_domain_randomization(domain_rand_cfg, sim_cfg.timestep)
  noise_cfg = go2_cfg.noise
  noise_scales = noise_cfg.noise_scales
  reward_cfg = go2_cfg.reward
  reward_scales = reward_cfg.scales

  # ---------- 传感器 ----------

  terrain_scan = RayCastSensorCfg(
    name = "terrain_scan",
    frame = ObjRef(type = "body", name = "", entity = "robot"),  # 按具体机器人设置。
    ray_alignment = "yaw",
    pattern = GridPatternCfg(
      size = tuple(sensor_cfg.terrain_scan_size),
      resolution = sensor_cfg.terrain_scan_resolution,
    ),
    max_distance = sensor_cfg.terrain_scan_max_distance,
    exclude_parent_body = True,
    debug_vis = sensor_cfg.terrain_scan_debug_vis,
    viz = RayCastSensorCfg.VizCfg(show_normals = sensor_cfg.terrain_scan_show_normals),
  )

  # ---------- 观测 ----------

  actor_terms = {
    "base_ang_vel": ObservationTermCfg(
      func = mdp.builtin_sensor,
      params = {"sensor_name": "robot/imu_ang_vel"},
      noise = Unoise(
        n_min = -noise_cfg.noise_level * noise_scales.base_ang_vel,
        n_max = noise_cfg.noise_level * noise_scales.base_ang_vel,
      ),
    ),
    "projected_gravity": ObservationTermCfg(
      func = mdp.projected_gravity,
      noise = Unoise(
        n_min = -noise_cfg.noise_level * noise_scales.projected_gravity,
        n_max = noise_cfg.noise_level * noise_scales.projected_gravity,
      ),
    ),
    "command": ObservationTermCfg(
      func = mdp.generated_commands,
      params = {"command_name": "twist"},
    ),
    "phase": ObservationTermCfg(
      func = mdp.phase,
      params = {
        "period": reward_cfg.gait_period,
        "command_name": "twist",
        "command_threshold": reward_cfg.command_threshold,
      },
    ),
    "joint_pos": ObservationTermCfg(
      func = mdp.joint_pos_rel,
      noise = Unoise(
        n_min = -noise_cfg.noise_level * noise_scales.joint_pos,
        n_max = noise_cfg.noise_level * noise_scales.joint_pos,
      ),
    ),
    "joint_vel": ObservationTermCfg(
      func = mdp.joint_vel_rel,
      noise = Unoise(
        n_min = -noise_cfg.noise_level * noise_scales.joint_vel,
        n_max = noise_cfg.noise_level * noise_scales.joint_vel,
      ),
    ),
    "actions": ObservationTermCfg(func = mdp.last_action),
  }

  critic_terms = {
    **deepcopy(actor_terms),
    "base_lin_vel": ObservationTermCfg(
      func = mdp.builtin_sensor,
      params = {"sensor_name": "robot/imu_lin_vel"},
      noise = Unoise(
        n_min = -noise_cfg.noise_level * noise_scales.base_lin_vel,
        n_max = noise_cfg.noise_level * noise_scales.base_lin_vel,
      ),
    ),
    "height_scan": ObservationTermCfg(
      func = envs_mdp.height_scan,
      params = {"sensor_name": "terrain_scan"},
      scale = 1 / terrain_scan.max_distance,
    ),
    "foot_height": ObservationTermCfg(
      func = mdp.foot_height,
      params = {"asset_cfg": SceneEntityCfg("robot", site_names = ())},  # 按具体机器人设置。
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

  # 只让 actor 感知固定标定误差；critic 始终读取仿真真值。
  actor_terms["joint_pos"].params["biased"] = True
  critic_terms["joint_pos"].params["biased"] = False
  actor_terms["base_ang_vel"].func = mdp.imu_ang_vel_with_errors
  actor_terms["projected_gravity"].func = mdp.imu_projected_gravity_with_errors

  observations = {
    "actor": ObservationGroupCfg(
      terms = actor_terms,
      concatenate_terms = True,
      enable_corruption = noise_cfg.add_noise,
      history_length = env_cfg.num_obs_history,
    ),
    "critic": ObservationGroupCfg(
      terms = critic_terms,
      concatenate_terms = True,
      enable_corruption = False,
      history_length = env_cfg.num_critic_obs_history,
    ),
  }

  # ---------- 指标 ----------

  metrics = {
    "mean_action_acc": MetricsTermCfg(
      func = mdp.mean_action_acc,
    ),
  }

  # ---------- 动作 ----------

  base_action_scale = control_cfg.action_scale
  scale = {
    r".*_hip_joint": base_action_scale * control_cfg.hip_reduction,
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

  def _build_commands() -> dict[str, CommandTermCfg]:
    ranges = UniformVelocityCommandCfg.Ranges(
      lin_vel_x = tuple(command_cfg.ranges.lin_vel_x),
      lin_vel_y = tuple(command_cfg.ranges.lin_vel_y),
      ang_vel_z = tuple(command_cfg.ranges.ang_vel_yaw),
      heading = tuple(command_cfg.ranges.heading),
    )
    for axis in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
      lower, upper = getattr(ranges, axis)
      if not all(math.isfinite(value) for value in (lower, upper)) or lower > upper:
        raise ValueError(f"Invalid target velocity range for {axis}.")
    if command_cfg.curriculum:
      initial = command_cfg.initial_ranges.lin_vel_x
      if not all(math.isfinite(value) for value in initial) or initial[0] > initial[1]:
        raise ValueError("Invalid initial velocity range for lin_vel_x.")
      lower, upper = ranges.lin_vel_x
      ranges.lin_vel_x = tuple(max(lower, min(upper, value)) for value in initial)
    return {
      "twist": UniformVelocityCommandCfg(
        entity_name = "robot",
        resampling_time_range = tuple(command_cfg.resampling_time),
        rel_standing_envs = command_cfg.rel_standing_envs,
        heading_command = command_cfg.heading_command,
        heading_control_stiffness = command_cfg.heading_control_stiffness,
        debug_vis = command_cfg.debug_vis,
        ranges = ranges,
      ),
    }

  # ---------- 事件 ----------

  events = {
    # 重置
    "reset_base": EventTermCfg(
      func = mdp.reset_root_state_uniform,
      mode = "reset",
      params = deepcopy(go2_cfg.reset.base_offset),
    ),
    "reset_robot_joints": EventTermCfg(
      func = mdp.reset_joints_by_offset,
      mode = "reset",
      params = {
        "position_range": tuple(go2_cfg.reset.joint_offset["position_range"]),
        "velocity_range": tuple(go2_cfg.reset.joint_offset["velocity_range"]),
        "asset_cfg": SceneEntityCfg("robot", joint_names = (".*",)),
      },
    ),

    # 域随机化
    "push_robot": EventTermCfg(
      func = mdp.push_by_setting_velocity,
      mode = "interval",
      interval_range_s = tuple(domain_rand_cfg.push_interval_s),
      params = {
        "velocity_range": {
          name: tuple(bounds)
          for name, bounds in domain_rand_cfg.push_velocity_range.items()
        },
      },
    ),
    "foot_friction": EventTermCfg(
      mode = "startup",
      func = dr.geom_friction,
      params = {
        "asset_cfg": SceneEntityCfg("robot", geom_names = ()),  # 按具体机器人设置。
        "operation": "abs",
        "ranges": tuple(domain_rand_cfg.foot_friction_range),
        "shared_random": True,  # 所有足端几何体共享同一个摩擦系数。
      },
    ),
    "encoder_bias": EventTermCfg(
      mode = "startup",
      func = dr.encoder_bias,
      params = {
        "asset_cfg": SceneEntityCfg("robot"),
        "bias_range": tuple(domain_rand_cfg.encoder_bias_range),
      },
    ),
    "base_com": EventTermCfg(
      mode = "startup",
      func = dr.pseudo_inertia,
      params = {
        "asset_cfg": SceneEntityCfg("robot", body_names = ()),  # 按具体机器人设置。
        "alpha_range": tuple(0.5 * math.log(scale) for scale in domain_rand_cfg.base_mass_range),
        "t_range": tuple(domain_rand_cfg.base_com_range),
        "distribution": "uniform",
      },
    ),
    "link_mass": EventTermCfg(
      mode = "startup",
      func = dr.pseudo_inertia,
      params = {
        "asset_cfg": SceneEntityCfg("robot", body_names = ()),  # 腿部选择不得包含基座。
        "alpha_range": tuple(0.5 * math.log(scale) for scale in domain_rand_cfg.link_mass_range),
        "distribution": "uniform",
      },
    ),
    "pd_gains": EventTermCfg(
      mode = "startup",
      func = dr.pd_gains,
      params = {
        "asset_cfg": SceneEntityCfg("robot"),
        "operation": "scale",
        "kp_range": tuple(domain_rand_cfg.kp_range),
        "kd_range": tuple(domain_rand_cfg.kd_range),
      },
    ),
    "effort_limits": EventTermCfg(
      mode = "startup",
      func = dr.effort_limits,
      params = {
        "asset_cfg": SceneEntityCfg("robot"),
        "operation": "scale",
        "effort_limit_range": tuple(domain_rand_cfg.effort_limit_range),
      },
    ),
    "joint_friction": EventTermCfg(
      mode = "startup",
      func = dr.joint_friction,
      params = {
        "asset_cfg": SceneEntityCfg("robot", joint_names = (".*_joint",)),
        "operation": "abs",
        "ranges": tuple(domain_rand_cfg.joint_friction_range),
      },
    ),
    "joint_armature": EventTermCfg(
      mode = "startup",
      func = dr.joint_armature,
      params = {
        "asset_cfg": SceneEntityCfg("robot", joint_names = (".*_joint",)),
        "operation": "scale",
        "ranges": tuple(domain_rand_cfg.joint_armature_range),
      },
    ),
    "joint_damping": EventTermCfg(
      mode = "startup",
      func = dr.joint_damping,
      params = {
        "asset_cfg": SceneEntityCfg("robot", joint_names = (".*_joint",)),
        "operation": "abs",
        "ranges": tuple(domain_rand_cfg.joint_damping_range),
      },
    ),
    "imu_errors": EventTermCfg(
      mode = "startup",
      func = mdp.randomize_imu_errors,
      params = {
        "gyro_bias_range": tuple(domain_rand_cfg.imu_gyro_bias_range),
        "mount_angle_range": tuple(domain_rand_cfg.imu_mount_angle_range),
      },
    ),
    "external_force": EventTermCfg(
      mode = "step",
      func = envs_mdp.apply_body_impulse,
      params = {
        "asset_cfg": SceneEntityCfg("robot", body_names = ()),  # 按具体机器人设置。
        "force_range": tuple(domain_rand_cfg.external_force_range),
        "torque_range": (0.0, 0.0),
        "duration_s": tuple(domain_rand_cfg.external_force_duration_s),
        "cooldown_s": tuple(domain_rand_cfg.external_force_cooldown_s),
      },
    ),
    "foot_contact_softness": EventTermCfg(
      mode = "startup",
      func = mdp.randomize_foot_contact_softness,
      params = {
        "asset_cfg": SceneEntityCfg("robot", geom_names = ()),  # 按具体机器人设置。
        "time_constant_range": tuple(domain_rand_cfg.foot_contact_time_constant_range),
      },
    ),
  }

  # ---------- 奖励 ----------

  rewards = {
    "track_linear_velocity": RewardTermCfg(
      func = mdp.track_linear_velocity,
      weight = reward_scales.track_linear_velocity,
      params = {"command_name": "twist", "std": reward_cfg.tracking_linear_std},
    ),
    "track_angular_velocity": RewardTermCfg(
      func = mdp.track_angular_velocity,
      weight = reward_scales.track_angular_velocity,
      params = {"command_name": "twist", "std": reward_cfg.tracking_angular_std},
    ),
    "body_orientation_l2": RewardTermCfg(
      func = mdp.body_orientation_l2,
      weight = reward_scales.body_orientation_l2,
      params = {"asset_cfg": SceneEntityCfg("robot", body_names = ())},  # 按具体机器人设置。
    ),
    "body_ang_vel": RewardTermCfg(
      func = mdp.body_angular_velocity_penalty,
      weight = reward_scales.body_ang_vel,
      params = {"asset_cfg": SceneEntityCfg("robot", body_names = ())},  # 按具体机器人设置。
    ),
    "angular_momentum": RewardTermCfg(
      func = mdp.angular_momentum_penalty,
      weight = reward_scales.angular_momentum,
      params = {"sensor_name": "robot/root_angmom"},
    ),
    "is_terminated": RewardTermCfg(
      func = mdp.is_terminated,
      weight = reward_scales.is_terminated,
    ),
    "joint_acc_l2": RewardTermCfg(
      func = mdp.joint_acc_l2,
      weight = reward_scales.joint_acc_l2,
    ),
    "joint_pos_limits": RewardTermCfg(
      func = mdp.joint_pos_limits,
      weight = reward_scales.joint_pos_limits,
    ),
    "action_rate_l2": RewardTermCfg(
      func = mdp.action_rate_l2,
      weight = reward_scales.action_rate_l2,
    ),
    "foot_gait": RewardTermCfg(
      func = mdp.feet_gait,
      weight = reward_scales.foot_gait,
      params = {
        "period": reward_cfg.gait_period,
        "offset": list(reward_cfg.gait_offset),
        "threshold": reward_cfg.gait_stance_fraction,
        "command_threshold": reward_cfg.command_threshold,
        "command_name": "twist",
        "sensor_name": "feet_ground_contact",
      }
    ),
    "foot_clearance": RewardTermCfg(
      func = mdp.feet_clearance,
      weight = reward_scales.foot_clearance,
      params = {
        "target_height": reward_cfg.foot_clearance_height,
        "command_name": "twist",
        "command_threshold": reward_cfg.command_threshold,
        "asset_cfg": SceneEntityCfg("robot", site_names = ()),  # 按具体机器人设置。
      },
    ),
    "foot_slip": RewardTermCfg(
      func = mdp.feet_slip,
      weight = reward_scales.foot_slip,
      params = {
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "command_threshold": reward_cfg.command_threshold,
        "asset_cfg": SceneEntityCfg("robot", site_names = ()),  # 按具体机器人设置。
      },
    ),
    "soft_landing": RewardTermCfg(
      func = mdp.soft_landing,
      weight = reward_scales.soft_landing,
      params = {
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "command_threshold": reward_cfg.command_threshold,
      },
    ),
    "stand_still": RewardTermCfg(
      func = mdp.stand_still,
      weight = reward_scales.stand_still,
      params = {
        "command_name": "twist",
        "command_threshold": reward_cfg.command_threshold,
        "asset_cfg": SceneEntityCfg("robot", joint_names = ".*"),
      },
    ),
  }

  # ---------- 终止 ----------

  terminations = {
    "time_out": TerminationTermCfg(func = mdp.time_out, time_out = True),
    "fell_over": TerminationTermCfg(
      func = mdp.bad_orientation,
      params = {"limit_angle": reward_cfg.fall_angle},
    ),
  }

  # ---------- 课程 ----------

  def _build_curriculum() -> dict[str, CurriculumTermCfg]:
    curriculum = {}
    if terrain_cfg.generator["curriculum"]:
      curriculum["terrain_levels"] = CurriculumTermCfg(
        func = mdp.terrain_levels_vel,
        params = {
          "command_name": "twist",
          "promotion_fraction": terrain_cfg.curriculum_promotion_fraction,
          "min_commanded_distance": terrain_cfg.curriculum_min_commanded_distance,
        },
      )
    if command_cfg.curriculum:
      curriculum["command_vel"] = CurriculumTermCfg(
        func = mdp.RewardBasedVelocityCurriculum,
        params = {
          "command_name": "twist",
          "linear_threshold": command_cfg.curriculum_threshold,
          "linear_increment": command_cfg.curriculum_increment,
          "lin_vel_x_limit": tuple(command_cfg.ranges.lin_vel_x),
          "check_interval_s": command_cfg.curriculum_update_interval_s,
        },
      )
    return curriculum

  # ---------- 组装并返回 ----------

  return ManagerBasedRlEnvCfg(
    scene = SceneCfg(
      terrain = TerrainEntityCfg(
        terrain_type = "generator",
        terrain_generator = TerrainGeneratorCfg(**deepcopy(terrain_cfg.generator)),
        max_init_terrain_level = terrain_cfg.max_init_terrain_level,
      ),
      sensors = (terrain_scan,),
      num_envs = env_cfg.num_envs,
      extent = env_cfg.scene_extent,
    ),
    observations = observations,
    actions = actions,
    commands = _build_commands(),
    events = events,
    seed = env_cfg.seed,
    rewards = rewards,
    terminations = terminations,
    curriculum = _build_curriculum(),
    metrics = metrics,
    viewer = ViewerConfig(
      origin_type = ViewerConfig.OriginType.ASSET_BODY,
      entity_name = "robot",
      body_name = "",  # 按具体机器人设置。
      distance = viewer_cfg.distance,
      elevation = viewer_cfg.elevation,
      azimuth = viewer_cfg.azimuth,
    ),
    sim = SimulationCfg(
      nconmax = sim_cfg.nconmax,
      njmax = sim_cfg.njmax,
      contact_sensor_maxmatch = sim_cfg.contact_sensor_maxmatch,
      mujoco = MujocoCfg(
        timestep = sim_cfg.timestep,
        iterations = sim_cfg.iterations,
        ls_iterations = sim_cfg.ls_iterations,
        ccd_iterations = sim_cfg.ccd_iterations,
      ),
    ),
    decimation = control_cfg.decimation,
    episode_length_s = env_cfg.episode_length_s,
  )
