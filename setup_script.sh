sudo wget -O /etc/apt/preferences.d/cuda-repository-pin-600 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin &&
sudo wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb &&
sudo dpkg -i cuda-keyring_1.0-1_all.deb &&
sudo add-apt-repository "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /" &&
sudo apt update &&
sudo apt install cuda=11.3.1-1

sudo apt install python3-pip &&
sudo apt install python-is-python3

pip install -U -qq git+https://github.com/huggingface/diffusers.git &&
pip install -U -qq accelerate tensorboard transformers ftfy &&
pip install -U -qq "ipywidgets>=7,<8" &&
pip install -qq bitsandbytes &&
pip install -U --pre triton &&
pip install -q https://github.com/TheLastBen/fast-stable-diffusion/raw/main/precompiled/A100/xformers-0.0.13.dev0-py3-none-any.whl &&
pip install torchvision