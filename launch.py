import os
import re
import time
import json
import requests
from datetime import timedelta

root_path = "/home/studio-lab-user/content"
webui_path = f"{root_path}/wibu"
#=================================
#=================================
def gpu_available():
    try:
        GPUs = GPUtil.getGPUs()
        return len(GPUs) > 0
    except Exception as e:
        return False
#=================================
get_ipython().run_line_magic('cd', '{webui_path}')
if gpu_available():
    print(f"[+] Running with GPU...")
    get_ipython().system('COMMANDLINE_ARGS="--no-download-sd --listen --xformers --theme dark --enable-insecure-extension-access --disable-console-progressbars --no-half-vae --ngrok {ngrok_token}" python launch.py')
else:
    print(f"[+] Running with CPU...")
    get_ipython().system('COMMANDLINE_ARGS="--no-download-sd --skip-torch-cuda-test --theme dark --precision full --enable-insecure-extension-access --ngrok {ngrok_token} --no-half --use-cpu SD GFPGAN BSRGAN ESRGAN SCUNet CodeFormer --all" python launch.py')
