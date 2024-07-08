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

home_dir= "/home/studio-lab-user"
ui_path = f"{home_dir}/content"
webui_path = f"{ui_path}/wibu"
ext_path = f"{webui_path}/extensions"

if not os.path.exists(webui_path):
    print("⌚ Install Stable Diffusion", end='')
    with capture.capture_output() as cap:
        get_ipython().system('aria2c --console-log-level=error -c -x 16 -s 16 -k 1M https://huggingface.co/godtrex99/REPO/resolve/main/Files/dogi.zip -o repo.zip')
        get_ipython().system('unzip -q -o repo.zip -d {ui_path}')
        get_ipython().system('rm -rf repo.zip')
        get_ipython().run_line_magic('cd', '{home_dir}')
        os.environ["SAFETENSORS_FAST_GPU"]='1'
        os.environ["CUDA_MODULE_LOADING"]="LAZY"
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        os.environ["PYTHONWARNINGS"] = "ignore"
    del cap

    print(" is complete!")
else:
    print("🚀 All unpacked... Skip. ⚡")
    action = "Updating WebUI and Extensions"
    print(f"⌚️ {action}...", end='', flush=True)
    with capture.capture_output() as cap:
        get_ipython().system('git config --global user.email "you@example.com"')
        get_ipython().system('git config --global user.name "Your Name"')
        get_ipython().run_line_magic('cd', '{webui_path}')
        get_ipython().system('git reset --hard && git pull')
        get_ipython().run_line_magic('cd', '{ext_path}')
        get_ipython().system('git fetch origin && git pull')
        get_ipython().system('cd {webui_path}/repositories/stable-diffusion-stability-ai && git restore .')
    del cap
    print(f"\r✨ {action} Completed!")