import numpy as np
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.base_env import ActionTuple
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel

SHAPE = (6, 31, 28)  # channels, height, width (matches ChompAgent's buffer layout)


class ChompEnv:
    """build=None -> talk to the Unity Editor (press Play when asked).
    Otherwise build = path to the built game."""

    def __init__(self, build=None, run_id=0, time_scale=20.0, no_graphics=False,
                 worker_id=0, **reward_params):
        self.engine = EngineConfigurationChannel()
        self.params = EnvironmentParametersChannel()
        self.env = UnityEnvironment(file_name=build, worker_id=worker_id,
                                    no_graphics=no_graphics,
                                    side_channels=[self.engine, self.params])
        self.engine.set_configuration_parameters(time_scale=time_scale, width=640,
                                                 height=360, quality_level=0)
        self.params.set_float_parameter("run_id", float(run_id))
        for k, v in reward_params.items():      # e.g. death_penalty=-100
            self.params.set_float_parameter(k, float(v))
        self.env.reset()
        self.name = list(self.env.behavior_specs)[0]

    def set_param(self, key, value):
        self.params.set_float_parameter(key, float(value))

    def reset(self):
        self.env.reset()
        ds, _ = self.env.get_steps(self.name)
        return ds.obs[0][0].reshape(SHAPE).astype(np.uint8)

    def step(self, action):
        self.env.set_actions(
            self.name,
            ActionTuple(discrete=np.array([[action]], dtype=np.int32)))
        self.env.step()
        ds, ts = self.env.get_steps(self.name)
        if len(ts) > 0:  # episode ended
            obs = ts.obs[0][0].reshape(SHAPE).astype(np.uint8)
            interrupted = getattr(ts, "interrupted", None)
            trunc = bool(interrupted[0]) if interrupted is not None else False
            return obs, float(ts.reward[0]), True, trunc
        obs = ds.obs[0][0].reshape(SHAPE).astype(np.uint8)
        return obs, float(ds.reward[0]), False, False   # obs, reward, done, truncated

    def close(self):
        self.env.close()