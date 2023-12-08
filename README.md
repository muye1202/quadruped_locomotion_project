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
