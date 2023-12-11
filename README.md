# quadruped_locomotion_project
A training and deployment pipeline for visuomotor RL policy used to enhance the locomotion of Unitree Go1 robot dog.

Author: Guo Ye, Muye Jia
This work is based on the following two papers: 
    <a href="https://gmargo11.github.io/walk-these-ways/" target="_blank">
      <b> Walk these Ways: Tuning Robot Control for Generalization with Multiplicity of Behavior </b>
    </a>
and:
    <a href="https://arxiv.org/abs/2107.03996" target="_blank">
      <b> Learning Vision-Guided Quadrupedal Locomotion End-to-End with Cross-Modal Transformers </b>
    </a>

The simulator we used is Isaac Gym from Nvidia (Paper: https://arxiv.org/abs/2108.10470); the Isaac Gym terrain and utilities code in `go1_gym` is built upon the 
<a href=https://github.com/leggedrobotics/legged_gym target="_blank">
    <b> Isaac Gym Environments for Legged Robots </b>
</a>
by Nikita Rudin, Robotic Systems Lab, ETH Zurich (Paper: https://arxiv.org/abs/2109.11978); the training structure in `go1_gym_learn/ppo_cse/actor_critic.py` is built upon
walk-these-ways `actor_critic.py` code in the same folder; `go1_gym_learn/ppo_cse/ppo.py` builds upon the 
<a href=https://github.com/leggedrobotics/rsl_rl target="_blank">
    <b> RSL RL </b>
</a>
repo developed by Nikita at Robotic Systems Lab, ETH Zurich; the deployment code in `go1_gym_deploy` builds upon the work from walk-these-ways, we modified the `deploy_policy.py`, `deployment_runner.py` and files in `utils` folder.


## Results


https://github.com/muye1202/quadruped_locomotion_project/assets/112987403/141b1108-f9a1-4453-8ddd-ecb63490b022



https://github.com/muye1202/quadruped_locomotion_project/assets/112987403/a6ff6fba-e04c-4421-86f1-516f5260e7a7



https://github.com/muye1202/quadruped_locomotion_project/assets/112987403/b62f4d4e-560e-4a9b-b52e-d74e7db77644




## Training

### Install Required Packages

#### Install pytorch 1.10 with cuda-11.3:

```bash
pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

#### Install Isaac Gym

1. Download and install Isaac Gym Preview 4 from https://developer.nvidia.com/isaac-gym
2. unzip the file via:
    ```bash
    tar -xf IsaacGym_Preview_4_Package.tar.gz
    ```

3. now install the python package
    ```bash
    cd isaacgym/python && pip install -e .
    ```
4. Verify the installation by try running an example

    ```bash
    python examples/1080_balls_of_solitude.py
    ```
5. For troubleshooting check docs `isaacgym/docs/index.html`

#### Install the `go1_gym` package

In this repository, run `pip install -e .`

#### Install the `ml_logger` package

run  `pip install ml-logger`

#### Install HuggingFace Transformer package

run `pip install transformers`

### Train the Visual Policy with ViT

In the root folder, run `python scripts/train.py`, the default visual encoder is Vision Transformer.

The ViT given in the example has the following parameters:
`hidden_size=64, num_hidden_layers=1, num_attention_heads=4, intermediate_size=64, hidden_dropout_prob=0.2, image_size=64`
this is a minimal ViT but training still takes at least 9 GB of GPU memory on Nvidia RTX 8000.

### Deploy a trained model

If the Intel Realsense depth camera is mounted on the Unitree Go1 and neural network inference is being done on external computers, then LCM communication can be used to transfer depth images from depth camera to the external computers. To publish the depth images, run `python3 depth_lcm/camera_msgs_pub.py`, and to test if LCM communication is established successfully, run `python3 depth_lcm/camera_msgs_sub.py` in a new terminal and check whether images are being saved.

If the Intel Realsense depth camera and the computing device is connected directly, the images can be read without the use of LCM communication.

### Installing the Deployment Utility

The first step is to connect your development machine to the robot using ethernet. You should ping the robot to verify the connection: `ping 192.168.123.15` should return `x packets transmitted, x received, 0% packet loss`.

Once you have confirmed the robot is connected, run the following command on your computer to transfer files to the robot. The first time you run it, the script will download and transfer the zipped docker image for development on the robot (`deployment_image.tar`). This file is quite large (3.5GB), but it only needs to be downloaded and transferred once.

```
cd go1_gym_deploy/scripts && ./send_to_unitree.sh
```

Next, you will log onto the robot's onboard computer and install the docker environment. To enter the onboard computer, the command is:

```
ssh unitree@192.168.123.15
```

Now, run the following commands on the robot's onboard computer:

```
chmod +x installer/install_deployment_code.sh
cd ~/go1_gym/go1_gym_deploy/scripts
sudo ../installer/install_deployment_code.sh
```

The installer will automatically unzip and install the docker image containing the deployment environment. 

### Setting up LCM communication

The Unitree Go1 and the external computer need to be under the same sub-net, the static ip needs to be set everytime on the external computer by running `sudo ifconfig -v eth0 192.168.123.xxx`, replacing `xxx` with viable number between 1-255.

### Running the Controller  <a name="runcontroller"></a>

Place the robot into damping mode. The control sequence is: [L2+A], [L2+B], [L1+L2+START]. After this, the robot should sit on the ground and the joints should move freely. 

If the neural network policy is run on an external computer, the following two commands need to be run on the same machine.

First:
```
cd ~/go1_gym/go1_gym_deploy/unitree_legged_sdk_bin
sudo ./lcm_position
```

Open up a new terminal and run the following command.

Second:
```
cd ~/go1_gym/go1_gym_deploy/scripts
python3 deploy_policy.py
```

The robot will wait for you to press [R2], then calibrate, then wait for a second press of [R2] before running the control loop.
