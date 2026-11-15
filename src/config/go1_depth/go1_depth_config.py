# Copyright (c) 2026-2027 zh
"""
内容：
    集中定义 Go1 Depth 的资产、双帧相机、MGDP 编码器与 Parkour Latent 训练配置。
    Go1 的 PD 与动作尺度沿用 wmp_go1@65cf1afd，Armature 与动作延迟对齐 Go2；算法参考 InstinctMJ@03563ee。
"""

from dataclasses import dataclass
from pathlib import Path

from mjlab.sensor import CameraSensorCfg

from agents.depth_supermimic.config import DepthSuperMimicConfig
from src.config.base.base_config import BaseConfig
from src.config.go2.go2_reference_supermimic_config import Go2ReferenceCfg


class Go1DepthCfg(BaseConfig):
    """Go1 的唯一任务配置；继承现有粗糙地形和全部 22 项奖励，实例之间独立。"""

    algorithm = "depth_supermimic"

    @dataclass
    class algo(DepthSuperMimicConfig):
        """潜变量模型沿用 Parkour、视觉编码采用 MGDP；采集和模型预算采用指定 Go2 配方。"""

        replay_capacity: int = 500
        collect_steps: float = 2.5
        warmup_steps: int = 64
        warmup_iterations: int = 0
        wm_hidden: tuple[int, ...] = (1024, 1024, 1024)
        wm_window: int = 8
        wm_batch: int = 2048
        wm_updates: int = 5
        latent_windows: int = 2048
        latent_recon_images: int = 512
        latent_weight: float = 1.0
        latent_recon_weight: float = 1.0
        auxiliary_weight: float = 1.0
        imag_batch: int = 2048
        horizon: int = 32
        actor_experts: int = 4
        actor_hidden: tuple[int, ...] = (256, 128, 64)
        critic_hidden: tuple[int, ...] = (512, 256, 128)
        actor_optimizer: str = "muon_adam"
        wm_optimizer: str = "muon_adam"
        muon_weight_decay: float = 0.1
        learning_rate: float = 4e-4
        actor_lr_scale: float = 4.0
        actor_ramp_iterations: int = 1000
        lr_decay_iterations: int = 30000
        lr_final_frac: float = 0.03
        action_noise: float = 0.3
        collect_det_frac: float = 0.25
        standing_zero_noise: bool = True

    class env:
        """默认4096环境、每环境500步回放；三份双帧深度约11.719 GiB，GRU起点约0.488 GiB。"""

        num_envs = 4096
        num_actions = 12
        env_spacing = 3.0
        scene_extent = 2.0
        episode_length_s = 20.0
        seed = 0
        num_obs_history = 5
        history_size = num_obs_history
        num_observations = 47
        num_obs = num_obs_history * num_observations
        # heading反馈需要额外六维命令状态，与 task.spec 的实际输入一致。
        num_privileged_obs = 286
        num_critic_obs = 286

    class experiment:
        contact_source = "model"
        actor_phase = True  # 原45维末尾追加sin／cos，与步态奖励共用时钟及站立门控。

    class sim(Go2ReferenceCfg.sim):
        """沿用 Go2 仿真预算，控制周期为 0.02 秒。"""

    class control:
        """PD 与动作尺度借鉴 wmp_go1；Armature 和动作延迟采用本项目 Go2 配置。"""

        control_type = "P"
        stiffness = {"hip": 40.0, "thigh": 40.0, "calf": 40.0}  # N·m/rad。
        damping = {"hip": 1.0, "thigh": 1.0, "calf": 1.0}  # N·m·s/rad。
        # 现有 XML 和源 go1_new_2.urdf 的全部关节限幅；不套用旧参考 XML 的电机值。
        effort_limit = {"hip": 33.5, "thigh": 33.5, "calf": 33.5}  # N·m。
        armature = {"hip": 0.01, "thigh": 0.01, "calf": 0.01}  # kg·m²。
        # 目标角 = 默认角 + action_scales[关节组] * 动作；无额外髋关节缩放。
        action_scales = {"hip": 0.25, "thigh": 0.25, "calf": 0.25}  # rad。
        decimation = 4  # 控制周期 0.02 s，50 Hz。
        soft_joint_pos_limit_factor = 0.9
        # 沿用 Go2 的 MjLab 指令延迟，lag 和更新周期均以物理步计。
        delay_min_lag = 0
        delay_max_lag = 4
        delay_hold_prob = 0.5
        delay_update_period = 10

    class collision:
        """MuJoCo 接触材料沿用本项目 Go2 初值，保留 Go1 全身碰撞及自碰撞。"""

        foot_condim = 3
        body_condim = 1
        foot_priority = 1
        foot_friction = (0.6,)
        foot_solimp = (0.9, 0.95, 0.023)
        enable_self_collision = True

    class init_state:
        """wmp_go1 的 Go1 出生位姿；米、弧度，前后腿采用不同默认角。"""

        pos = [0.0, 0.0, 0.35]
        default_joint = {
            "FL_hip_joint": 0.1,
            "FL_thigh_joint": 0.8,
            "FL_calf_joint": -1.5,
            "FR_hip_joint": -0.1,
            "FR_thigh_joint": 0.8,
            "FR_calf_joint": -1.5,
            "RL_hip_joint": 0.1,
            "RL_thigh_joint": 1.0,
            "RL_calf_joint": -1.5,
            "RR_hip_joint": -0.1,
            "RR_thigh_joint": 1.0,
            "RR_calf_joint": -1.5,
        }

    class reset:
        """相对于所分配地块应用出生位姿及关节重置范围。"""

        base_offset = {
            "pose_range": {"x": (-0.5, 0.5),
                           "y": (-0.5, 0.5),
                           "z": (0.0, 0.0),
                           "roll": (0.0, 0.0),
                           "pitch": (0.0, 0.0),
                           "yaw": (-3.14, 3.14)},
            "velocity_range": {},
        }
        joint_offset = {"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0)}

    class sensors:
        """Parkour 深度处理、历史和训练专用扫描协议。"""

        contact_num_slots = 1
        nonfoot_history_length = 1

        class depth:
            """v3：原始32×32带噪深度、双帧历史和整数控制步延迟。"""

            protocol_version = 3
            camera_name = "depth_camera"
            width = 32
            height = 32
            data_types = ("depth",)
            enabled_geom_groups = (0, 3)
            use_textures = False
            use_shadows = False
            noise_gaussian = 0.03  # 米；直接对原始有效命中加噪，干净标签保持原值。
            noise_dropout = 0.1  # 独立像素置零概率；仅对带噪输入生效。
            near_clip = 0.1
            near_max_hops = 6
            near_epsilon = 1e-4
            far_clip = 2.5
            history_length = 2
            sensor_history_length = 8
            history_skip_frames = 5
            delay_range = (0, 2)  # 闭区间整数控制步；对应0、20、40毫秒。
            update_interval = 1  # 每个控制步 sense，不额外降频。

        class height_scan:
            """yaw 对齐，按参考 xy 网格展开；不输入 actor。"""

            measured_points_x = tuple(round(-0.8 + 0.1 * i, 2) for i in range(17))
            measured_points_y = tuple(round(-0.5 + 0.1 * i, 2) for i in range(11))
            max_height = 5.0

    class terrain(Go2ReferenceCfg.terrain):
        """直接使用 Go2 SuperMimic rough 地形及行走表现课程。"""

        mode = "rough"

    class reward(Go2ReferenceCfg.reward):
        """原样复用 Go2 的 22 项公式、权重及控制周期缩放，不新增奖励。"""

    class domain_randomization(Go2ReferenceCfg.domain_randomization):
        """沿用有上下文表示的摩擦、编码器偏置和质心随机化。"""

    class commands(Go2ReferenceCfg.commands):
        """继承 Go2 的前后、侧移、转向范围、heading反馈及命令重采样。"""

    class runner:
        """5000 轮训练、每 5000 轮保存；执行拆分不改变算法样本量。"""

        max_iterations = 5000
        save_interval = 5000
        log_interval = 50
        metric_interval = 50
        compile = True
        compile_backend = "inductor"
        fused_adam = True
        cuda_graph = True  # CUDA捕获模型、actor、整段critic及两类回放采样；CPU关闭。
        actor_bf16 = True
        encoder_bf16 = True
        wm_bf16 = True
        critic_bf16 = True
        decoder_bf16 = True
        imag_microbatch = 4096  # 与Go2一致；2048条想象默认整批完成。
        critic_microbatch = 32768
        replay_history_storage = "frames"  # 只压缩真实带噪本体历史；深度保持完整uint8栈。

    class noise:
        """统一控制真实本体与深度噪声；仅显式full回放模式重新采本体噪声。"""

        add_noise = True
        noise_level = 1.0

        class noise_scales:
            base_ang_vel = 0.2
            projected_gravity = 0.05
            joint_pos = 0.01
            joint_vel = 0.5

    class viewer:
        """沿用 Go2 跟随视角；距离为米，角度为度，不改变机载深度相机。"""

        distance = 1.5
        elevation = -10.0
        azimuth = 90.0

def make_go1_depth_camera_cfg(go1_cfg: Go1DepthCfg | None = None, *, entity_name: str = "robot") -> CameraSensorCfg:
    """装配已命名实体的深度传感器；挂载位姿和 VFOV 直接沿用 XML。"""
    cfg = go1_cfg if go1_cfg is not None else Go1DepthCfg()
    depth = cfg.sensors.depth
    return CameraSensorCfg(
        name="depth",
        camera_name=f"{entity_name}/{depth.camera_name}",
        width=depth.width,
        height=depth.height,
        data_types=tuple(depth.data_types),
        enabled_geom_groups=tuple(depth.enabled_geom_groups),
        use_textures=depth.use_textures,
        use_shadows=depth.use_shadows,
    )


@dataclass(frozen=True)
class Go1DepthTrainConfig:
    """入口只声明显式覆盖；训练默认统一取上方 Go1DepthCfg。"""

    device: str = "cuda:0"
    num_envs: int | None = None
    seed: int | None = None
    max_iterations: int | None = None
    save_interval: int | None = None
    log_interval: int | None = None
    compile: bool | None = None
    cuda_graph: bool | None = None
    imag_microbatch: int | None = None
    log_root: str = "logs/depth_supermimic"
    run_name: str = ""
    task_config: Path | None = None
    resume: Path | None = None
