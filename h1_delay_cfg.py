"""H1 rough task + control-latency randomization. Only change vs Isaac-Velocity-Rough-H1-v0: actuators become explicit PD
(DelayedPDActuator) whose targets reach the joints 0-40 ms late, a new random lag per episode. Same Kp/Kd/torque limits.
Explicit PD at the 5 ms physics step is also what sim2sim.py does in MuJoCo.
"""
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.h1.rough_env_cfg import H1RoughEnvCfg

MAX_DELAY_STEPS = 8  # x 5 ms physics step = 40 ms


@configclass
class H1RoughDelayEnvCfg(H1RoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot.actuators = {
            k: DelayedPDActuatorCfg(joint_names_expr=a.joint_names_expr, effort_limit=a.effort_limit_sim,
                                    stiffness=a.stiffness, damping=a.damping, min_delay=0, max_delay=MAX_DELAY_STEPS)
            for k, a in self.scene.robot.actuators.items()
        }
