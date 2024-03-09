#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import os
import re
import time
import json
import requests
import subprocess
from datetime import timedelta
from subprocess import getoutput
from urllib.parse import unquote
from IPython.utils import capture
from IPython.display import clear_output


#  ================= DETECT ENV =================
root_path = "/home/studio-lab-user/content"
webui_path = f"{root_path}/wibu"
get_ipython().system('mkdir -p {root_path}')
# Check if gpu or cpu
output = getoutput('nvidia-smi --query-gpu=gpu_name --format=csv')
if "name" in output:
    gpu_name = output[5:]
    clear_output()
    print(f'\r✅ \033[92;1mCurrent GPU:\033[0;1m {gpu_name}\033[97;1m', flush=True)
else:
    clear_output()
    print('\r\033[91;1m❎ ERROR: GPU - not detected. \nThe startup will be performed on the CPU.\n\033[97;1m', flush=True)
#  ----------------------------------------------
print("Updating dependencies, may take some time...")
get_ipython().system('pip install -q --upgrade torchsde')
get_ipython().system('pip install -q --upgrade pip')
get_ipython().system('pip install -q --upgrade psutil')
clear_output()


# ================ LIBRARIES ================
flag_file = f"{root_path}/libraries_installed.txt"

if not os.path.exists(flag_file):
    xformers = "xformers==0.0.23.post1 triton==2.1.0"
    torch = "torch==2.1.2+cu121 torchvision==0.16.2+cu121 --extra-index-url https://download.pytorch.org/whl/cu121"
    print("Installing the libraries, it's going to take a while....", end='')
    with capture.capture_output() as cap:
        get_ipython().run_line_magic('pip', 'install -q opencv-python-headless huggingface-hub')
        get_ipython().run_line_magic('conda', 'update -q -n base conda')
        get_ipython().run_line_magic('conda', 'install -q -y aria2')
        get_ipython().run_line_magic('conda', 'install -q -y glib')
        get_ipython().system('pip install -q {torch} -U')
        get_ipython().system('pip install -q {xformers} -U')
        get_ipython().system('pip install tensorflow')
        get_ipython().system('pip install gputil')
        get_ipython().system('wget -P /home/studio-lab-user https://huggingface.co/NagisaNao/fast_repo/resolve/main/sagemaker/FULL_DELETED_NOTEBOOK.ipynb')
        with open(flag_file, "w") as f:
            f.write("hey ;3")
    del cap
    print("\rLibraries are installed!" + " "*35)
    time.sleep(2)
    clear_output()

# CONFIG DIR
models_dir = f"{webui_path}/models/Stable-diffusion"
vaes_dir = f"{webui_path}/models/VAE"
embeddings_dir = f"{webui_path}/embeddings"
loras_dir = f"{webui_path}/models/Lora"
extensions_dir = f"{webui_path}/extensions"
control_dir = f"{webui_path}/models/ControlNet"

# ================= MAIN CODE =================
# --- Obsolescence warning ---
if not os.path.exists(webui_path):
    start_install = int(time.time())
    print("⌚ Unpacking Stable Diffusion...", end='')
    with capture.capture_output() as cap:
        get_ipython().system('aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/godtrex99/REPO/resolve/main/Files/UI.zip -o repo.zip')
        get_ipython().system('unzip -q -o repo.zip -d {root_path}')
        get_ipython().system('rm -rf repo.zip')
        get_ipython().run_line_magic('cd', '{root_path}')
        os.environ["SAFETENSORS_FAST_GPU"]='1'
        os.environ["CUDA_MODULE_LOADING"]="LAZY"
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        os.environ["PYTHONWARNINGS"] = "ignore"
    del cap

    print("🚀 Unpacking is complete!")
else:
    print("🚀 All unpacked... Skip. ⚡")
    action = "Updating WebUI and Extensions"
    print(f"⌚️ {action}...", end='', flush=True)
    with capture.capture_output() as cap:
        get_ipython().system('git config --global user.email "you@example.com"')
        get_ipython().system('git config --global user.name "Your Name"')
        get_ipython().system('{\'for dir in \' + webui_path + \'/extensions/*/; do cd \\"$dir\\" && git reset --hard && git pull; done\'}')
        get_ipython().system('{\'for dir in /home/studio-lab-user/content/wibu/extensions/*/; do cd \\"$dir\\" && git fetch origin && git pull; done\'}')
        get_ipython().system('cd {webui_path}/repositories/stable-diffusion-stability-ai && git restore .')
        get_ipython().system('wget -O {webui_path}/modules/styles.py https://huggingface.co/NagisaNao/fast_repo/resolve/main/sagemaker/fixing/webui/styles.py')
    del cap
    print(f"\r✨ {action} Completed!")

# Cleaning shit after downloading...
get_ipython().system('find  \\( -name ".ipynb_checkpoints" -o -name ".aria2" \\) -type d -exec rm -r {} \\; >/dev/null 2>&1')

## List Models and stuff
if any(not file.endswith('.txt') for file in os.listdir(models_dir)):
    print("\n\033[33m➤ Models\033[0m")
    get_ipython().system("find {models_dir}/ -mindepth 1 ! -name '*.txt' -printf '%f\\n'")
if any(not file.endswith('.txt') for file in os.listdir(vaes_dir)):
    print("\n\033[33m➤ VAEs\033[0m")
    get_ipython().system("find {vaes_dir}/ -mindepth 1 ! -name '*.txt' -printf '%f\\n'")
if any(not file.endswith('.txt') and not os.path.isdir(os.path.join(embeddings_dir, file)) for file in os.listdir(embeddings_dir)):
    print("\n\033[33m➤ Embeddings\033[0m")
    get_ipython().system("find {embeddings_dir}/ -mindepth 1 -maxdepth 1 \\( -name '*.pt' -or -name '*.safetensors' \\) -printf '%f\\n'")
if any(not file.endswith('.txt') for file in os.listdir(loras_dir)):
    print("\n\033[33m➤ LoRAs\033[0m")
    get_ipython().system("find {loras_dir}/ -mindepth 1 ! -name '*.keep' -printf '%f\\n'")
print(f"\n\033[33m➤ Extensions\033[0m")
get_ipython().system("find {extensions_dir}/ -mindepth 1 -maxdepth 1 ! -name '*.txt' -printf '%f\\n'")
if any(not file.endswith(('.txt', '.yaml')) for file in os.listdir(control_dir)):
    print("\n\033[33m➤ ControlNet\033[0m")
    get_ipython().system("find {control_dir}/ -mindepth 1 ! -name '*.yaml' -printf '%f\\n' | sed 's/^[^_]*_[^_]*_[^_]*_\\(.*\\)_fp16\\.safetensors$/\\1/'")

with capture.capture_output() as cap:
    get_ipython().system('rm -rf /home/studio-lab-user/.conda/envs/studiolab-safemode')
    get_ipython().system('rm -rf /home/studio-lab-user/.conda/envs/sagemaker-distribution')
    get_ipython().system('rm -rf /home/studio-lab-user/.conda/pkgs/cache')
    get_ipython().system('pip cache purge')
    get_ipython().system('rm -rf ~/.cache')
del cap

