import torch
import torch.nn as nn
from params_proto import PrefixProto
from torch.distributions import Normal

from go1_gym_learn.ppo_cse.conv2d import Conv2dHeadModel, DepthEncoder
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class TransformerActor(nn.Module):
    """Simple Transformer-based actor network."""

    def __init__(self, input_dim, num_actions, nhead=4, num_layers=2):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(
            d_model=input_dim,
            nhead=nhead,
            batch_first=True,
            activation="relu",
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_dim, num_actions)

    def forward(self, x):
        # x shape: (batch, input_dim)
        x = x.unsqueeze(1)  # add sequence dimension
        x = self.transformer(x)
        x = x.squeeze(1)
        return self.fc(x)


class AC_Args(PrefixProto, cli=False):
    # policy
    init_noise_std = 1.0
    actor_hidden_dims = [512, 256, 128]
    critic_hidden_dims = [512, 256, 128]
    activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid

    actor_model = 'mlp'  # 'mlp' or 'transformer'
    transformer_nhead = 4
    transformer_num_layers = 2

    adaptation_module_branch_hidden_dims = [256, 128]

    use_decoder = False


class ActorCritic(nn.Module):
    is_recurrent = False

    def __init__(self, num_obs,
                 num_privileged_obs,
                 num_obs_history,
                 num_actions,
                 **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str(
                [key for key in kwargs.keys()]))
        self.decoder = AC_Args.use_decoder
        super().__init__()

        self.num_obs_history = num_obs_history
        self.num_privileged_obs = num_privileged_obs

        activation = get_activation(AC_Args.activation)

        # Adaptation module
        self.num_depth_latent = 64
        adaptation_module_layers = []
        adaptation_module_layers.append(nn.Linear(self.num_obs_history + self.num_depth_latent, AC_Args.adaptation_module_branch_hidden_dims[0]))
        adaptation_module_layers.append(activation)
        for l in range(len(AC_Args.adaptation_module_branch_hidden_dims)):
            if l == len(AC_Args.adaptation_module_branch_hidden_dims) - 1:
                adaptation_module_layers.append(
                    nn.Linear(AC_Args.adaptation_module_branch_hidden_dims[l], self.num_privileged_obs))
            else:
                adaptation_module_layers.append(
                    nn.Linear(AC_Args.adaptation_module_branch_hidden_dims[l],
                              AC_Args.adaptation_module_branch_hidden_dims[l + 1]))
                adaptation_module_layers.append(activation)
        self.adaptation_module = nn.Sequential(*adaptation_module_layers)

        # Policy
        actor_input_dim = self.num_privileged_obs + self.num_obs_history
        if AC_Args.actor_model == 'transformer':
            self.actor_body = TransformerActor(
                actor_input_dim,
                num_actions,
                nhead=AC_Args.transformer_nhead,
                num_layers=AC_Args.transformer_num_layers,
            )
            actor_desc = "Actor Transformer"
        else:
            actor_layers = []
            actor_layers.append(nn.Linear(actor_input_dim, AC_Args.actor_hidden_dims[0]))
            actor_layers.append(activation)
            for l in range(len(AC_Args.actor_hidden_dims)):
                if l == len(AC_Args.actor_hidden_dims) - 1:
                    actor_layers.append(nn.Linear(AC_Args.actor_hidden_dims[l], num_actions))
                else:
                    actor_layers.append(nn.Linear(AC_Args.actor_hidden_dims[l], AC_Args.actor_hidden_dims[l + 1]))
                    actor_layers.append(activation)
            self.actor_body = nn.Sequential(*actor_layers)
            actor_desc = "Actor MLP"

        # Value function
        critic_layers = []
        critic_layers.append(nn.Linear(self.num_privileged_obs + self.num_obs_history, AC_Args.critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(AC_Args.critic_hidden_dims)):
            if l == len(AC_Args.critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(AC_Args.critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(AC_Args.critic_hidden_dims[l], AC_Args.critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
        self.critic_body = nn.Sequential(*critic_layers)

        print(f"Adaptation Module: {self.adaptation_module}")
        print(f"{actor_desc}: {self.actor_body}")
        print(f"Critic MLP: {self.critic_body}")

        # Action noise
        self.std = nn.Parameter(AC_Args.init_noise_std * torch.ones(num_actions))
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False

        #Guo
        self.visual_obs_slice = slice(70, 3142, None), (1, 48, 64)
        self.visual_latent_size = 256
        self.visual_kwargs = dict(
            channels= [64, 64],
            kernel_sizes= [3, 3],
            strides= [1, 1],
            hidden_sizes= [256],
        )

        self.visual_encoder = Conv2dHeadModel(
            image_shape= self.visual_obs_slice[1],
            output_size= self.visual_latent_size,
            **self.visual_kwargs,
        )

        self.depth_encoder = DepthEncoder()

    @staticmethod
    # not used at the moment
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observation_history, depth_obs):
        #Guo, add depth latent
        depth_latent = self.depth_encoder(depth_obs)
        concat_adaptation_input = torch.cat((observation_history, depth_latent),dim=-1)
        latent = self.adaptation_module(concat_adaptation_input)

        mean = self.actor_body(torch.cat((observation_history, latent), dim=-1))
        self.distribution = Normal(mean, mean * 0. + self.std)

    def act(self, observation_history, depth_obs, **kwargs):
        self.update_distribution(observation_history, depth_obs)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_expert(self, ob, policy_info={}):
        return self.act_teacher(ob["obs_history"], ob["privileged_obs"])

    def act_inference(self, ob, policy_info={}):
        return self.act_student(ob["obs_history"], policy_info=policy_info)

    def act_student(self, observation_history, depth_observations, policy_info={}):
        depth_latent = self.depth_encoder(depth_observations)
        concat_adaptation_input = torch.cat((observation_history, depth_latent),dim=-1)
        latent = self.adaptation_module(concat_adaptation_input)
        actions_mean = self.actor_body(torch.cat((observation_history, latent), dim=-1))
        policy_info["latents"] = latent.detach().cpu().numpy()
        return actions_mean

    def act_teacher(self, observation_history, privileged_info, policy_info={}):
        print("use teacher")
        actions_mean = self.actor_body(torch.cat((observation_history, privileged_info), dim=-1))
        policy_info["latents"] = privileged_info
        return actions_mean

    def evaluate(self, observation_history, privileged_observations, depth_observations, **kwargs):
        # depth_latent = self.depth_encoder(depth_observations)
        value = self.critic_body(torch.cat((observation_history, privileged_observations), dim=-1))
        return value

    def get_student_latent(self, observation_history):
        return self.adaptation_module(observation_history)

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
