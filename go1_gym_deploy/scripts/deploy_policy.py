import glob
import pickle as pkl
import lcm
import sys
sys.path.append("/home/nano/walk-these-ways/")
from go1_gym_deploy.utils.deployment_runner import DeploymentRunner
from go1_gym_deploy.envs.lcm_agent import LCMAgent
from go1_gym_deploy.utils.cheetah_state_estimator import StateEstimator
from go1_gym_deploy.utils.command_profile import *
from go1_gym_learn.ppo_cse.actor_critic import ActorCritic
from transformers import ViTConfig, ViTModel
import pathlib

lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=254")

def load_and_run_policy(label, depth, experiment_name, max_vel=1.0, max_yaw_vel=1.0):
    # load agent
    dirs = glob.glob(f"../../runs/{label}/*")
    logdir = sorted(dirs)[0]

    with open(logdir+"/parameters.pkl", 'rb') as file:
        pkl_cfg = pkl.load(file)
        # print(pkl_cfg.keys())
        cfg = pkl_cfg["Cfg"]
        # print(cfg.keys())


    se = StateEstimator(lc)

    control_dt = 0.02
    command_profile = RCControllerProfile(dt=control_dt, state_estimator=se, x_scale=max_vel, y_scale=0.6, yaw_scale=max_yaw_vel)

    hardware_agent = LCMAgent(cfg, se, command_profile)
    se.spin()

    sys.path.append("/home/nano/walk-these-ways/go1_gym_deploy")
    from envs.history_wrapper import HistoryWrapper
    hardware_agent = HistoryWrapper(hardware_agent)

    policy, depth_encoder = load_policy(logdir, depth)   # Muye

    # load runner
    root = f"{pathlib.Path(__file__).parent.resolve()}/../../logs/"
    pathlib.Path(root).mkdir(parents=True, exist_ok=True)
    deployment_runner = DeploymentRunner(lcm_connection=lc, se=None, log_root=f"{root}/{experiment_name}")
    deployment_runner.add_control_agent(hardware_agent, "hardware_closed_loop")
    deployment_runner.add_policy(policy)
    if depth_encoder is not None:
        deployment_runner.add_encoder(depth_encoder)   # Muye
    deployment_runner.add_command_profile(command_profile)

    if len(sys.argv) >= 2:
        max_steps = int(sys.argv[1])
    else:
        max_steps = 10000

    deployment_runner.run(max_steps=max_steps, logging=True)
    # deployment_runner._test_recv()   # Muye
    # deployment_runner.run_depth(max_steps=max_steps)   # Muye

def load_policy(logdir, depth=True):
    actor_critic = ActorCritic(70, 2, 2100, 12)
    ac_weights = torch.load(logdir + '/checkpoints/ac_weights_last.pt')
    actor_critic.load_state_dict(state_dict=ac_weights) # torch.jit.load(logdir + '/body_latest.jit').cuda()
    body = actor_critic.actor_body.cuda()

    # # Load HuggingFace ViT
    # model_id = logdir
    # config = ViTConfig.from_pretrained(model_id)
    # vit_model = ViTModel.from_pretrained(model_id, config=config)

    import os
    adaptation_module = torch.jit.load(logdir + '/checkpoints/adaptation_module_latest.jit').cuda()
    if depth:
        depth_encoder = torch.jit.load(logdir + '/checkpoints/depth_encoder_module_latest.jit').cuda()
        # depth_encoder = vit_model.cuda()

    def depth_policy(obs, depth_obs, info):
        depth_latent = depth_encoder.forward(depth_obs.cuda()) # depth_encoder(depth_obs.cuda())[1]
        concat_input = torch.cat((obs["obs_history"].cuda(), depth_latent.cuda()), dim=-1)
        latent = adaptation_module.forward(concat_input)
        action = body.forward(torch.cat((obs["obs_history"].cuda(), latent.cuda()), dim=-1))
        info['latent'] = latent

        return action

    def policy(obs, info):
        i = 0
        latent = adaptation_module.forward(obs["obs_history"].to('cpu'))
        action = body.forward(torch.cat((obs["obs_history"].to('cpu'), latent), dim=-1))
        info['latent'] = latent
        return action

    if depth:
        return depth_policy, depth_encoder

    return policy, None


if __name__ == '__main__':
    label = "gait-conditioned-agility/pretrain-v0/train"
    depth_label = "gait-conditioned-agility/depth_0929/train" #"gait-conditioned-agility/vit_depth"

    experiment_name = "example_experiment"

    load_and_run_policy(depth_label, depth=True, experiment_name=experiment_name, max_vel=3.5, max_yaw_vel=5.0)
    # load_and_run_policy(label, depth=False, experiment_name=experiment_name, max_vel=3.5, max_yaw_vel=5.0)
