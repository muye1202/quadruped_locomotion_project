FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04

ENV TZ=UTC \
    DEBIAN_FRONTEND=noninteractive \
    CARB_DISABLE_PLUGINS=libnvf.plugin.so \
    ISAAC_GYM_ENABLE_VIEWER=0

# Timezone + base deps + Vulkan SDK bits
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone \
 && apt-get update --allow-insecure-repositories \
 && apt-get install -y --allow-unauthenticated --no-install-recommends \
      python3.8 python3.8-dev python3-pip python3.8-venv git \
      build-essential libcurl4-openssl-dev libssl-dev \
      libvulkan1 libvulkan-dev vulkan-tools vulkan-utils \
      vulkan-validationlayers libgl1-mesa-glx libx11-6 \
      mesa-vulkan-drivers \
 && rm -rf /var/lib/apt/lists/*

# Python setup
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 \
 && python -m pip install --upgrade pip \
 && python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 \
 && python -m pip install ml_logger

# Install IsaacGym
COPY IsaacGym_Preview_4_Package /workspace/IsaacGym
WORKDIR /workspace/IsaacGym/isaacgym/python
RUN python -m pip install --no-cache-dir -e .

WORKDIR /workspace/project
ENTRYPOINT ["bash"]
