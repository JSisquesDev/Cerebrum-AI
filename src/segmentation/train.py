from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from dotenv import load_dotenv
from src.segmentation.model import create_unet
from src.segmentation.data import load_data, get_training_files, create_dataframe, split_dataframe

import tensorflow as tf
import pandas as pd

import os
import time
import math 

def train(model, train_gen, steps_per_epoch, validation_data, epochs, callbacks, validation_steps):
    
    # Iniciamos el crono
    start = time.time()
    
    # Entrenamos el modelo
    result = model.fit(
        train_gen,
        steps_per_epoch = steps_per_epoch, 
        epochs = epochs,
        callbacks = callbacks,
        validation_data = validation_data,
        validation_steps = validation_steps
    )
    
    # Paramos el crono
    end = time.time() - start

    # Obtenemos el tiempo en minutos
    train_time = end / 60

    # Mostramos el tiempo total de entreno
    print(f"Tiempo total de entrenamiento: {train_time} minutos")
    
    return result

if __name__ == '__main__':
    # Cargamos las variables de entorno para la detección
    ENV = load_dotenv(f'.{os.sep}config{os.sep}.env.segmentation')
    
    # Comprobamos que se use la GPU
    print(f"Dispositivo de entrenamiento: {tf.config.list_physical_devices('GPU')}")
    
    # Configuramos TensorFlow
    os.environ["TF_DIRECTML_MAX_ALLOC_SIZE"] = "536870912" # 512MB
    os.environ["TF_CPP_MIN_VLOG_LEVEL"] = "3"
    
    # Establecemos las variables comunes a los modelos
    OUTPUT_PATH = str(os.getenv("MODEL_PATH"))
    DATASET_PATH = str(os.getenv("DATASET_PATH"))
    RESTORE_BEST_WEIGHTS = bool(os.getenv("RESTORE_BEST_WEIGHTS"))
    
    # Cargamos las variables para el modelo UNET
    model_path = f"{OUTPUT_PATH}{os.sep}{str(os.getenv('UNET_MODEL_NAME'))}{os.sep}{str(os.getenv('UNET_MODEL_NAME'))}"
    checkpoint_path = f"{OUTPUT_PATH}{os.sep}{str(os.getenv('UNET_MODEL_NAME'))}{os.sep}checkpoint{os.sep}checkpoint.h5"
    batch_size = int(os.getenv("UNET_BATCH_SIZE"))
    epochs = int(os.getenv("UNET_EPOCHS"))
    patience = int(os.getenv("UNET_PATIENT"))
    img_height = int(os.getenv("UNET_IMG_HEIGHT"))
    img_width = int(os.getenv("UNET_IMG_WIDTH"))
    img_deep = int(os.getenv("UNET_IMG_DEEP"))
    activation = str(os.getenv("UNET_ACTIVATION"))
    image_color_mode = str(os.getenv("UNET_IMAGE_COLOR_MODE"))
    mask_color_mode = str(os.getenv("UNET_MASK_COLOR_MODE"))
    image_save_prefix = str(os.getenv("UNET_IMAGE_SAVE_PREFIX"))
    mask_save_prefix = str(os.getenv("UNET_MASK_SAVE_PREFIX"))
    
    # Obtenemos los ficheros para entrenar al modelo
    train_files, mask_files = get_training_files(DATASET_PATH)
    
    # Creamos el dataframe
    dataframe = create_dataframe(
        train_files=train_files, 
        mask_files=mask_files
    )
    
    # Dividimos el dataframe en train, validation y test
    df_train, df_val, df_test = split_dataframe(dataframe)
    
    # Cargamos los datos para UNET
    train_gen, test_gen = load_data(
        dataframe = dataframe,
        target_size = (img_height, img_width),
        batch_size = batch_size,
        image_color_mode = image_color_mode,
        image_save_prefix = image_save_prefix,
        mask_color_mode = mask_color_mode,
        mask_save_prefix = mask_save_prefix,
        seed = 1
    )
    
    # Generamos el modelo UNET
    unet = create_unet(img_height, img_width, img_deep, activation, epochs)
    
    # Generamos los callbacks para UNET
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=RESTORE_BEST_WEIGHTS)
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_binary_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks=[early_stopping, checkpoint]
 
    # Entrenamos el modelo UNET
    history = train(
        model = unet,
        train_gen = train_gen,
        steps_per_epoch = len(df_train) / batch_size,
        validation_data = test_gen,
        epochs = epochs,
        callbacks = callbacks,
        validation_steps = len(df_val) / batch_size
    )
    
    # Guardamos el modelo Unet
    acc = math.floor(history.history["binary_accuracy"][-1] * 100)
    unet.save(f'{model_path}_{acc}.h5', save_format='tf')
    
    # Creamos un dataframe con el historial del entrenamiento y lo guardamos
    history_dataframe = pd.DataFrame(history.history)
    history_dataframe.to_csv(f'{model_path}_{acc}.csv', index=False, sep=str(os.getenv('SEPARATOR')))