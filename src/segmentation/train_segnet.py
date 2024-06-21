from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from dotenv import load_dotenv
from src.segmentation.model import create_segnet
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
    
    # Cargamos las variables para el modelo Segnet
    model_path = f"{OUTPUT_PATH}{os.sep}{str(os.getenv('SEGNET_MODEL_NAME'))}{os.sep}{str(os.getenv('SEGNET_MODEL_NAME'))}"
    checkpoint_path = f"{OUTPUT_PATH}{os.sep}{str(os.getenv('SEGNET_MODEL_NAME'))}{os.sep}checkpoint{os.sep}checkpoint.h5"
    batch_size = int(os.getenv("SEGNET_BATCH_SIZE"))
    epochs = int(os.getenv("SEGNET_EPOCHS"))
    patience = int(os.getenv("SEGNET_PATIENT"))
    img_height = int(os.getenv("SEGNET_IMG_HEIGHT"))
    img_width = int(os.getenv("SEGNET_IMG_WIDTH"))
    img_deep = int(os.getenv("SEGNET_IMG_DEEP"))
    activation = str(os.getenv("SEGNET_ACTIVATION"))
    image_color_mode = str(os.getenv("SEGNET_IMAGE_COLOR_MODE"))
    mask_color_mode = str(os.getenv("SEGNET_MASK_COLOR_MODE"))
    image_save_prefix = str(os.getenv("SEGNET_IMAGE_SAVE_PREFIX"))
    mask_save_prefix = str(os.getenv("SEGNET_MASK_SAVE_PREFIX"))
    
    # Obtenemos los ficheros para entrenar al modelo
    train_files, mask_files = get_training_files(DATASET_PATH)
    
    # Creamos el dataframe
    dataframe = create_dataframe(
        train_files=train_files, 
        mask_files=mask_files
    )
    
    # Dividimos el dataframe en train, validation y test
    df_train, df_val, df_test = split_dataframe(dataframe)
    
    # Cargamos los para entrenar a Segnet
    train_gen, train_val_gen = load_data(
        dataframe = df_train,
        target_size = (img_height, img_width),
        batch_size = batch_size,
        image_color_mode = image_color_mode,
        image_save_prefix = image_save_prefix,
        mask_color_mode = mask_color_mode,
        mask_save_prefix = mask_save_prefix,
        seed = 1
    )
    
    # Cargamos los para validar a Segnet
    test_gen, test_val_gen = load_data(
        dataframe = df_train,
        target_size = (img_height, img_width),
        batch_size = batch_size,
        image_color_mode = image_color_mode,
        image_save_prefix = image_save_prefix,
        mask_color_mode = mask_color_mode,
        mask_save_prefix = mask_save_prefix,
        seed = 1
    )
    
    # Generamos el modelo Segnet
    segnet = create_segnet(img_height, img_width, img_deep, activation, epochs)
    
    # Generamos los callbacks para Segnet
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=RESTORE_BEST_WEIGHTS)
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='dice_coef', verbose=1, save_best_only=True, mode='max')
    callbacks=[early_stopping, checkpoint]
 
    # Entrenamos el modelo Segnet
    history = train(
        model = segnet,
        train_gen = train_gen,
        steps_per_epoch = len(df_train) / batch_size,
        validation_data = train_val_gen,
        epochs = epochs,
        callbacks = callbacks,
        validation_steps = len(df_val) / batch_size
    )
    
    # Evaluamos el modelo Segnet
    evaluation = segnet.evaluate(
        x = test_val_gen,
        batch_size=batch_size,
        verbose="auto",
        steps = len(df_test) / batch_size,
        return_dict = True
    )
    
    # Obtenemos los resultados de la evaluación
    eval_loss = math.floor(evaluation['loss'] * 100)
    eval_binary_accuracy = math.floor(evaluation['binary_accuracy'] * 100) 
    eval_iou = math.floor(evaluation['iou'] * 100) 
    eval_dice_coef = math.floor(evaluation['dice_coef'] * 100)
    
    # Creamos un sufijo con los valores
    sufix = f'l{eval_loss}_ba{eval_binary_accuracy}_iou{eval_iou}_dc{eval_dice_coef}'
    
    # Creamos el dataframe de la evaluación y lo guardamos
    evaluation_dataframe = pd.DataFrame.from_dict([evaluation])
    evaluation_dataframe.to_csv(f'{model_path}_evaluation_{sufix}.csv', index=False, sep=str(os.getenv('SEPARATOR')))
    
    # Guardamos el modelo Segnet
    segnet.save(f'{model_path}_{sufix}.h5', save_format='tf')
    
    # Creamos un dataframe con el historial del entrenamiento y lo guardamos
    history_dataframe = pd.DataFrame(history.history)
    history_dataframe.to_csv(f'{model_path}_history_{sufix}.csv', index=False, sep=str(os.getenv('SEPARATOR')))