sudo wget -O /etc/apt/preferences.d/cuda-repository-pin-600 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin &&
sudo wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb &&
sudo dpkg -i cuda-keyring_1.0-1_all.deb &&
sudo add-apt-repository "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /" &&
sudo apt update &&
sudo apt install cuda=11.3.1-1

find / -name libcuda.so 2>/dev/null

curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

sudo apt install python3-pip &&
sudo apt install python-is-python3

conda install -y -c pytorch -c conda-forge cudatoolkit=11.3 pytorch torchvision torchaudio

pip install -U git+https://github.com/huggingface/diffusers.git &&
pip install accelerate tensorboard transformers ftfy &&
pip install "ipywidgets>=7,<8" &&
pip install bitsandbytes &&
pip install -U --pre triton &&
pip install -q https://github.com/TheLastBen/fast-stable-diffusion/raw/main/precompiled/A100/xformers-0.0.13.dev0-py3-none-any.whl &&
pip install torchvision

git clone https://github.com/pschinis/dreambooth-setup.git &&
cd dreambooth-setup &&
conda env create -f environment.yaml && conda activate dbt &&
python sd_dreambooth_training.py

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-11.3/targets/x86_64-linux/lib/stubs/libcuda.so
pip3 install torch==1.12.0+cu113 torchvision==0.13.0+cu113 -f https://download.pytorch.org/whl/torch_stable.htmlcd dreambooth

Setup to get mismatched types error:
1) Plain Ubuntu VM
2) Sudo install python
3) Install Cuda with Google script
4) Pip install up to ipywidgets + torchvision
5) Run script without unet to cuda or bitsandbytes

With full precision, no bits and bites, triton error: 15:30
With half precision: 9:00