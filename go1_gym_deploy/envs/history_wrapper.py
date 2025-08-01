import torch


class HistoryWrapper:
    def __init__(self, env):
        self.env = env

        if isinstance(self.env.cfg, dict):
            self.obs_history_length = self.env.cfg["env"]["num_observation_history"]
        else:
            self.obs_history_length = self.env.cfg.env.num_observation_history
        self.num_obs_history = self.obs_history_length * self.env.num_obs
        self.obs_history = torch.zeros(self.env.num_envs, self.num_obs_history, dtype=torch.float,
                                       device=self.env.device, requires_grad=False)
        self.num_privileged_obs = self.env.num_privileged_obs

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        privileged_obs = info["privileged_obs"]
        depth_obs = info.get("depth_obs")   # Muye
        if depth_obs is None:
            if getattr(self.env, 'depth_images', None) is not None:
                depth_obs = self.env.depth_images
            else:
                h = getattr(self.env.cfg.env, 'depth_camera_height_px', 1) if not isinstance(self.env.cfg, dict) else self.env.cfg['env'].get('depth_camera_height_px', 1)
                w = getattr(self.env.cfg.env, 'depth_camera_width_px', 1) if not isinstance(self.env.cfg, dict) else self.env.cfg['env'].get('depth_camera_width_px', 1)
                depth_obs = torch.zeros(self.env.num_envs, 1, h, w,
                                        device=self.env.device,
                                        dtype=torch.float)

        self.obs_history = torch.cat((self.obs_history[:, self.env.num_obs:], obs), dim=-1)
        return {'obs': obs, 'privileged_obs': privileged_obs, 'depth_obs': depth_obs,
                'obs_history': self.obs_history}, rew, done, info

    def get_observations(self):
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        depth_obs = self.env.depth_images if getattr(self.env, 'depth_images', None) is not None else None
        if getattr(self.env, 'depth_images', None) is not None:
            depth_obs = self.env.depth_images
        else:
            if isinstance(self.env.cfg, dict):
                h = self.env.cfg['env'].get('depth_camera_height_px', 1)
                w = self.env.cfg['env'].get('depth_camera_width_px', 1)
            else:
                h = getattr(self.env.cfg.env, 'depth_camera_height_px', 1)
                w = getattr(self.env.cfg.env, 'depth_camera_width_px', 1)
            depth_obs = torch.zeros(self.env.num_envs, 1, h, w,
                                    device=self.env.device,
                                    dtype=torch.float)

        self.obs_history = torch.cat((self.obs_history[:, self.env.num_obs:], obs), dim=-1)
        return {'obs': obs, 'privileged_obs': privileged_obs, 
                'depth_obs': depth_obs, 'obs_history': self.obs_history}   # Muye

    def get_obs(self):
        obs = self.env.get_obs()
        privileged_obs = self.env.get_privileged_observations()
        depth_obs = self.env.depth_images if getattr(self.env, 'depth_images', None) is not None else None
        if getattr(self.env, 'depth_images', None) is not None:
            depth_obs = self.env.depth_images
        else:
            if isinstance(self.env.cfg, dict):
                h = self.env.cfg['env'].get('depth_camera_height_px', 1)
                w = self.env.cfg['env'].get('depth_camera_width_px', 1)
            else:
                h = getattr(self.env.cfg.env, 'depth_camera_height_px', 1)
                w = getattr(self.env.cfg.env, 'depth_camera_width_px', 1)
            depth_obs = torch.zeros(self.env.num_envs, 1, h, w,
                                    device=self.env.device,
                                    dtype=torch.float)

        self.obs_history = torch.cat((self.obs_history[:, self.env.num_obs:], obs), dim=-1)
        return {'obs': obs, 'privileged_obs': privileged_obs, 
                'depth_obs': depth_obs, 'obs_history': self.obs_history}

    def reset_idx(self, env_ids):  # it might be a problem that this isn't getting called!!
        ret = self.env.reset_idx(env_ids)
        self.obs_history[env_ids, :] = 0
        return ret

    def reset(self):
        ret = self.env.reset()
        privileged_obs = self.env.get_privileged_observations()
        depth_obs = self.env.depth_images if getattr(self.env, 'depth_images', None) is not None else None
        if getattr(self.env, 'depth_images', None) is not None:
            depth_obs = self.env.depth_images
        else:
            if isinstance(self.env.cfg, dict):
                h = self.env.cfg['env'].get('depth_camera_height_px', 1)
                w = self.env.cfg['env'].get('depth_camera_width_px', 1)
            else:
                h = getattr(self.env.cfg.env, 'depth_camera_height_px', 1)
                w = getattr(self.env.cfg.env, 'depth_camera_width_px', 1)
            depth_obs = torch.zeros(self.env.num_envs, 1, h, w,
                                    device=self.env.device,
                                    dtype=torch.float)

        self.obs_history[:, :] = 0
        return {"obs": ret, "privileged_obs": privileged_obs, 
                "depth_obs": depth_obs, "obs_history": self.obs_history}

    def __getattr__(self, name):
        return getattr(self.env, name)
