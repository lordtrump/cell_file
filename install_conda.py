import os
import json
import subprocess

home = os.path.expanduser("~")
os.makedirs(os.path.join(home, "content/special"), exist_ok=True)
#os.makedirs(os.path.join(home, "content/files_cells"), exist_ok=True)

def update_conda():
    # startup init default
    user_home_dir = "/home/studio-lab-user"
    
    update_tasks = [
        ("conda install -y conda glib psutil gperftools aria2 gdown nodejs", "Installing Conda", "0"),  # Clear
        ("conda install -y -n base python=3.10.12", "Installing Python 3.10", "33"),  # Yellow
        ("conda clean -y --all", "Cleaning Conda", "32"),  # Green
        ("pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 torchaudio==2.1.2 --extra-index-url https://download.pytorch.org/whl/cu121", "Installing Torch", "33"),  # Yellow
        ("pip install xformers==0.0.23.post1 triton==2.1.0", "Installing xformers", "33")  # Yellow
    ]

    update_info = {}

    for command, message, color in update_tasks:
        print(f"\033[1;{color}m{message}\033[0m")
        subprocess.run(command, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        update_info[message] = "✔️"

    # Dir save file JSON
    directory_path = os.path.expanduser("~/content/special")
    json_file_path = os.path.join(directory_path, "update.json")

    with open(json_file_path, "w") as file:
        json.dump(update_info, file)
        
    print("\033[1;0mDone\033[0m")
    
    get_ipython().kernel.do_shutdown(True)

def check_updates():
    json_file_path = os.path.expanduser("~/content/special/update.json")
    if os.path.exists(json_file_path):
        with open(json_file_path, "r") as file:
            updates = json.load(file)
        print("All components are updated:")
        for component, status in updates.items():
            print(f"{component}: {status}")
    else:
        update_conda()

check_updates()