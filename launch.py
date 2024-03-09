import os
from subprocess import getoutput
from IPython.display import clear_output
output = getoutput('nvidia-smi --query-gpu=gpu_name --format=csv')
root_path = "/home/studio-lab-user/content"
webui_path = f"{root_path}/wibu"
get_ipython().run_line_magic('cd', '{webui_path}')
if "name" in output:
    gpu_name = output[5:]
    clear_output()
    print(f'\r✅ \033[92;1mCurrent GPU:\033[0;1m {gpu_name}\033[97;1m', flush=True)
    get_ipython().system('COMMANDLINE_ARGS="--no-download-sd --listen --xformers --theme dark --enable-insecure-extension-access --disable-console-progressbars --no-half-vae --ngrok {ngrok_token}" python launch.py')
else:
    clear_output()
    print('\r\033[91;1m❎ ERROR: GPU - not detected. \nThe startup will be performed on the CPU.\n\033[97;1m', flush=True)
    get_ipython().system('COMMANDLINE_ARGS="--no-download-sd --skip-torch-cuda-test --theme dark --precision full --enable-insecure-extension-access --ngrok {ngrok_token} --no-half --use-cpu SD GFPGAN BSRGAN ESRGAN SCUNet CodeFormer --all" python launch.py')
