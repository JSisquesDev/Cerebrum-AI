from dotenv import load_dotenv

import os
import sys

if __name__ == '__main__':

    # Cargamos las variables de entorno
    load_dotenv(f'.{os.sep}config{os.sep}.env.segmentation')
    
    # Obtenemos los argumentos de ejecución
    args = sys.argv
    
    # En el caso de que no se den argumentos se entrena todo
    train_all = args.__len__() == 1

    if train_all:
        print(f'Entrenando el modelo U-Net...')
        os.system('python -m src.segmentation.train_unet')
        
        print(f'Entrenando el modelo Segnet...')
        os.system('python -m src.segmentation.train_segnet')
    else:
        for arg in args[1:]:
            if arg == str(os.getenv('UNET_MODEL_NAME')):
                print(f'Entrenando el modelo U-Net...')
                os.system('python -m src.segmentation.train_unet')
            elif arg == str(os.getenv('SEGNET_MODEL_NAME')):
                print(f'Entrenando el modelo Segnet...')
                os.system('python -m src.segmentation.train_segnet')
            else:
                print(f'No existe ningun modelo llamado {arg}')
            