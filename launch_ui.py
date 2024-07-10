import os
import re
import time
import json
import requests
from datetime import timedelta
from subprocess import getoutput
from IPython.display import clear_output

try:
  start_colab
except:
  start_colab = int(time.time())-5

output = getoutput('nvidia-smi --query-gpu=gpu_name --format=csv')
home_dir= "/home/studio-lab-user"
ui_path = f"{home_dir}/content"
webui_path = f"{ui_path}/wibu"

get_ipython().run_line_magic('cd', '{webui_path}')
get_ipython().system('echo -n {start_colab} > {webui_path}/static/colabTimer.txt')

if "name" in output:
    gpu_name = output[5:]
    clear_output()
    print(f'\r✅ \033[92;1mCurrent GPU:\033[0;1m {gpu_name}\033[97;1m', flush=True)
    get_ipython().system('COMMANDLINE_ARGS="--no-download-sd --listen --xformers --theme dark --enable-insecure-extension-access --disable-console-progressbars --no-half-vae --ngrok {ngrok_token}" python launch.py')
    clear_output()
    start_colab = float(open(f'{webui_path}/static/colabTimer.txt', 'r').read())
    time_since_start = str(timedelta(seconds=time.time()-start_colab)).split('.')[0]
    print(f"\n⌚️ \033[0mВы Waktu Yang Kamu Habiskan Selama - \033[33m{time_since_start}\033[0m\n\n")
else:
    clear_output()
    print('\r\033[91;1m❎ ERROR: GPU - not detected. \nThe startup will be performed on the CPU.\n\033[97;1m', flush=True)
    get_ipython().system('COMMANDLINE_ARGS="--no-download-sd --skip-torch-cuda-test --theme dark --precision full --enable-insecure-extension-access --ngrok {ngrok_token} --no-half --use-cpu SD GFPGAN BSRGAN ESRGAN SCUNet CodeFormer" python launch.py')
    clear_output()
    start_colab = float(open(f'{webui_path}/static/colabTimer.txt', 'r').read())
    time_since_start = str(timedelta(seconds=time.time()-start_colab)).split('.')[0]
    print(f"\n⌚️ \033[0mВы Waktu Yang Kamu Habiskan Selama - \033[33m{time_since_start}\033[0m\n\n")
    