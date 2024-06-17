import os
import shutil

# Actualizamos pip
os.system("python -m pip install --upgrade pip")

# Instalamos los requerimientos
os.system("pip3 install -r requirements.txt")

# Guardamos la ruta absoluta el proyecto
path = os.path.abspath("./")
os.putenv("PROJECT_PATH", path)

env_variables = {}
env_file_path = './config/.env'

with open(env_file_path, "r") as f:
    for line in f.readlines():
        key, value = line.split('=')
        env_variables[key] = value

if "PROJECT_PATH" not in env_variables:
    with open(env_file_path, "w") as f:
        for key in env_variables:
            f.write(f'{key}={env_variables[key]}')
        f.write(f'PROJECT_PATH={path}\n')
        
# Descargamos los datasets
os.system("python download_datasets.py")