import copy
import time
import os
import sys
sys.path.append("/home/nano/walk-these-ways/")
from go1_gym_deploy.lcm_types.camera_message_lcmt import camera_message_lcmt
import numpy as np
import torch
import threading
from depth_tests.realsense import Go1RealSense
from go1_gym_deploy.utils.logger import MultiLogger


class DeploymentRunner:
    def __init__(self, lcm_connection, se=None, log_root="."):
        self.agents = {}
        self.policy = None
        self.command_profile = None
        self.logger = MultiLogger()
        self.se = se
        self.vision_server = None

        self.log_root = log_root
        self.init_log_filename()
        self.control_agent_name = None
        self.command_agent_name = None

        self.triggered_commands = {i: None for i in range(4)} # command profiles for each action button on the controller
        self.button_states = np.zeros(4)

        self.is_currently_probing = False
        self.is_currently_logging = [False, False, False, False]

        # Muye --> Depth Image module
        if lcm_connection is not None:
            self.lc = lcm_connection
            self.depth_sub = self.lc.subscribe("depth_image", self.realsense_callback)

        self.depth_image = None
        self.depth_image_init = torch.ones((1,1,48,64))
        self.depth_encoder = None
        self.agent = None
        self.depth_action = None
        self.img_list = []

    # Muye --> realsense callback
    #############################
    def realsense_callback(self, channel, msg):
        img_str = camera_message_lcmt.decode(msg)
        utf_str = img_str.data
        img = np.frombuffer(utf_str, dtype=np.uint16)
        img = np.array(img/5000.0, dtype=np.float32)
        img_tensor = torch.from_numpy(img).float().view(1,1,64,64)
        img_resized = torch.nn.functional.interpolate(img_tensor, size=(48, 64))

        self.depth_image = img_resized

    def _test_recv(self):
       while self.depth_image is None:
         print("waiting for image")

       if self.depth_image is not None:
         print(f"depth image shape: {self.depth_image.shape}")

    def add_encoder(self, depth_enc):
        """
        Add the depth encoder if using vision
        """
        self.depth_encoder = depth_enc

    # process depth encoder in a separate thread
    def process_depth(self, control_obs, policy_info):
        while self.depth_image is not None:
            self.lc.handle()
            action = self.policy(control_obs, self.depth_image, policy_info)
            self.depth_action = action

    #############################

    def init_log_filename(self):
        datetime = time.strftime("%Y/%m_%d/%H_%M_%S")

        for i in range(100):
            try:
                os.makedirs(f"{self.log_root}/{datetime}_{i}")
                self.log_filename = f"{self.log_root}/{datetime}_{i}/log.pkl"
                return
            except FileExistsError:
                continue


    def add_open_loop_agent(self, agent, name):
        self.agents[name] = agent
        self.logger.add_robot(name, agent.env.cfg)

    def add_control_agent(self, agent, name):
        self.control_agent_name = name
        self.agents[name] = agent
        self.logger.add_robot(name, agent.env.cfg)

    def add_vision_server(self, vision_server):
        self.vision_server = vision_server

    def set_command_agents(self, name):
        self.command_agent = name

    def add_policy(self, policy):
        self.policy = policy

    def add_command_profile(self, command_profile):
        self.command_profile = command_profile

    def calibrate(self, wait=True, low=False):
        # first, if the robot is not in nominal pose, move slowly to the nominal pose
        for agent_name in self.agents.keys():
            if hasattr(self.agents[agent_name], "get_obs"):
                agent = self.agents[agent_name]
                agent.get_obs()
                joint_pos = agent.dof_pos
                if low:
                    final_goal = np.array([0., 0.3, -0.7,
                                           0., 0.3, -0.7,
                                           0., 0.3, -0.7,
                                           0., 0.3, -0.7,])
                else:
                    final_goal = np.zeros(12)
                nominal_joint_pos = agent.default_dof_pos

                print(f"About to calibrate; the robot will stand [Press R2 to calibrate]")
                while wait:
                    self.button_states = self.command_profile.get_buttons()
                    if self.command_profile.state_estimator.right_lower_right_switch_pressed:
                        self.command_profile.state_estimator.right_lower_right_switch_pressed = False
                        break

                cal_action = np.zeros((agent.num_envs, agent.num_actions))
                target_sequence = []
                target = joint_pos - nominal_joint_pos
                while np.max(np.abs(target - final_goal)) > 0.01:
                    target -= np.clip((target - final_goal), -0.05, 0.05)
                    target_sequence += [copy.deepcopy(target)]
                for target in target_sequence:
                    next_target = target
                    if isinstance(agent.cfg, dict):
                        hip_reduction = agent.cfg["control"]["hip_scale_reduction"]
                        action_scale = agent.cfg["control"]["action_scale"]
                    else:
                        hip_reduction = agent.cfg.control.hip_scale_reduction
                        action_scale = agent.cfg.control.action_scale

                    next_target[[0, 3, 6, 9]] /= hip_reduction
                    next_target = next_target / action_scale
                    cal_action[:, 0:12] = next_target
                    agent.step(torch.from_numpy(cal_action))
                    agent.get_obs()
                    time.sleep(0.05)

                print("Starting pose calibrated [Press R2 to start controller]")
                while True:
                    self.button_states = self.command_profile.get_buttons()
                    if self.command_profile.state_estimator.right_lower_right_switch_pressed:
                        self.command_profile.state_estimator.right_lower_right_switch_pressed = False
                        break

                for agent_name in self.agents.keys():
                    obs = self.agents[agent_name].reset()
                    if agent_name == self.control_agent_name:
                        control_obs = obs

        return control_obs


    def run(self, num_log_steps=1000000000, max_steps=100000000, logging=True):
        assert self.control_agent_name is not None, "cannot deploy, runner has no control agent!"
        assert self.policy is not None, "cannot deploy, runner has no policy!"
        assert self.command_profile is not None, "cannot deploy, runner has no command profile!"

        # TODO: add basic test for comms

        for agent_name in self.agents.keys():
            obs = self.agents[agent_name].reset()
            if agent_name == self.control_agent_name:
                control_obs = obs

        control_obs = self.calibrate(wait=True)

        # now, run control loop
        policy_info = {}
        if self.depth_encoder is not None:
            cam_obj = Go1RealSense()
            cam_obj.start_depth_thread()

            while cam_obj.get_latest_frame() is None:
                print("depth frame has not yet arrived")

        try:
            for i in range(max_steps):
                policy_info = {}

                # Muye
                if self.depth_encoder is not None:
                    latest_depth_frame = cam_obj.get_latest_frame()
                    # print(f"latest frame {latest_depth_frame}")
                    start_time = time.time()
                    if latest_depth_frame is not None:
                        action = self.policy(control_obs, latest_depth_frame, policy_info)
                        end_time = time.time()
                        print(f"depth action: {action}")
                        print(f"time used: {end_time - start_time}")
                    else:
                        print("depth frame is empty")
                else:
                    start_time = time.time()
                    print("using state-only actions")
                    action = self.policy(control_obs, policy_info)
                    end_time = time.time()
                    print(f"time used: {end_time - start_time}")

                for agent_name in self.agents.keys():
                    obs, ret, done, info = self.agents[agent_name].step(action)

                    info.update(policy_info)
                    info.update({"observation": obs, "reward": ret, "done": done, "timestep": i,
                                 "time": i * self.agents[self.control_agent_name].dt, "action": action, "rpy": self.agents[self.control_agent_name].se.get_rpy(), "torques": self.agents[self.control_agent_name].torques})

                    if logging: self.logger.log(agent_name, info)

                    if agent_name == self.control_agent_name:
                        control_obs, control_ret, control_done, control_info = obs, ret, done, info

                # bad orientation emergency stop
                rpy = self.agents[self.control_agent_name].se.get_rpy()
                if abs(rpy[0]) > 1.6 or abs(rpy[1]) > 1.6:
                    self.calibrate(wait=False, low=True)

                # check for logging command
                prev_button_states = self.button_states[:]
                self.button_states = self.command_profile.get_buttons()

                if self.command_profile.state_estimator.left_lower_left_switch_pressed:
                    if not self.is_currently_probing:
                        print("START LOGGING")
                        self.is_currently_probing = True
                        self.agents[self.control_agent_name].set_probing(True)
                        self.init_log_filename()
                        self.logger.reset()
                    else:
                        print("SAVE LOG")
                        self.is_currently_probing = False
                        self.agents[self.control_agent_name].set_probing(False)
                        # calibrate, log, and then resume control
                        control_obs = self.calibrate(wait=False)
                        self.logger.save(self.log_filename)
                        self.init_log_filename()
                        self.logger.reset()
                        time.sleep(1)
                        control_obs = self.agents[self.control_agent_name].reset()
                    self.command_profile.state_estimator.left_lower_left_switch_pressed = False

                for button in range(4):
                    if self.command_profile.currently_triggered[button]:
                        if not self.is_currently_logging[button]:
                            print("START LOGGING")
                            self.is_currently_logging[button] = True
                            self.init_log_filename()
                            self.logger.reset()
                    else:
                        if self.is_currently_logging[button]:
                            print("SAVE LOG")
                            self.is_currently_logging[button] = False
                            # calibrate, log, and then resume control
                            control_obs = self.calibrate(wait=False)
                            self.logger.save(self.log_filename)
                            self.init_log_filename()
                            self.logger.reset()
                            time.sleep(1)
                            control_obs = self.agents[self.control_agent_name].reset()

                if self.command_profile.state_estimator.right_lower_right_switch_pressed:
                    control_obs = self.calibrate(wait=False)
                    time.sleep(1)
                    self.command_profile.state_estimator.right_lower_right_switch_pressed = False
                    # self.button_states = self.command_profile.get_buttons()
                    while not self.command_profile.state_estimator.right_lower_right_switch_pressed:
                        time.sleep(0.01)
                        # self.button_states = self.command_profile.get_buttons()
                    self.command_profile.state_estimator.right_lower_right_switch_pressed = False

            # finally, return to the nominal pose
            control_obs = self.calibrate(wait=False)
            self.logger.save(self.log_filename)

        except KeyboardInterrupt:
            self.logger.save(self.log_filename)
