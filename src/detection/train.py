from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from dotenv import load_dotenv
from src.detection.model import create_alexnet, create_vgg19
from src.detection.data import load_data

import tensorflow as tf

import os
import time

def train(model, train_data, validation_data, epochs, callbacks):
    
    # Iniciamos el crono
    start = time.time()
    
    # Entrenamos el modelo
    result = model.fit(train_data, epochs=epochs, callbacks=callbacks, validation_data=validation_data)
    
    # Paramos el crono
    end = time.time() - start

    # Obtenemos el tiempo en minutos
    train_time = end / 60

    # Mostramos el tiempo total de entreno
    print(f"Tiempo total de entrenamiento: {train_time} minutos")
    
    return result

if __name__ == '__main__':
    # Cargamos las variables de entorno para la detección
    ENV = load_dotenv(f'.{os.sep}config{os.sep}.env.detection')
    
    # Comprobamos que se use la GPU
    print(f"Dispositivo de entrenamiento: {tf.config.list_physical_devices('GPU')}")
    
    # Establecemos las variables comunes a los modelos
    OUTPUT_PATH = str(os.getenv("MODEL_PATH"))
    DATASET_PATH = str(os.getenv("DATASET_PATH"))
    RESTORE_BEST_WEIGHTS = bool(os.getenv("RESTORE_BEST_WEIGHTS"))
    ACTIVATION = str(os.getenv('ACTIVATION'))
    COLOR_MODE = str(os.getenv('COLOR_MODE'))
    CLASS_MODE = str(os.getenv('CLASS_MODE'))
    '''
    # Cargamos las variables para el modelo AlexNet
    model_path = f"{OUTPUT_PATH}{os.sep}{str(os.getenv('ALEXNET_MODEL_NAME'))}{os.sep}{str(os.getenv('ALEXNET_MODEL_NAME'))}"
    checkpoint_path = f"{OUTPUT_PATH}{os.sep}{str(os.getenv('ALEXNET_MODEL_NAME'))}{os.sep}checkpoint{os.sep}checkpoint.h5"
    batch_size = int(os.getenv("ALEXNET_BATCH_SIZE"))
    epochs = int(os.getenv("ALEXNET_EPOCHS"))
    patience = int(os.getenv("ALEXNET_PATIENT"))
    img_height = int(os.getenv("ALEXNET_IMG_HEIGHT"))
    img_width = int(os.getenv("ALEXNET_IMG_WIDTH"))
    img_deep = int(os.getenv("ALEXNET_IMG_DEEP"))
    
    # Cargamos los datos para AlexNet
    train_data, validation_data = load_data(DATASET_PATH, (img_height, img_width), batch_size, COLOR_MODE, CLASS_MODE)
    
    # Obtenemos las etiquetas y el número de categorias
    labels = train_data.class_indices
    num_categories = labels.__len__() - 1
    
    # Generamos el modelo Alexnet
    alexnet = create_alexnet(img_height, img_width, img_deep, num_categories, ACTIVATION)
    
    # Generamos los callbacks para Alexnet
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=RESTORE_BEST_WEIGHTS)
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks=[early_stopping, checkpoint]
 
    # Entrenamos el modelo AlexNet
    history = train(
        model = alexnet,
        train_data = train_data,
        validation_data = validation_data,
        epochs = epochs,
        callbacks = callbacks
    )
    
    # Guardamos el modelo AlexNet con el valor accuracy
    acc = round(history.history["accuracy"][-1] * 100)
    alexnet.save(f'{model_path}_{acc}.h5', save_format='tf')
    '''
    
    # Cargamos las variables para el modelo VGG19
    model_path = f"{OUTPUT_PATH}{os.sep}{str(os.getenv('VGG19_MODEL_NAME'))}{os.sep}{str(os.getenv('VGG19_MODEL_NAME'))}"
    checkpoint_path = f"{OUTPUT_PATH}{os.sep}{str(os.getenv('VGG19_MODEL_NAME'))}{os.sep}checkpoint{os.sep}checkpoint.h5"
    batch_size = int(os.getenv("VGG19_BATCH_SIZE"))
    epochs = int(os.getenv("VGG19_EPOCHS"))
    patience = int(os.getenv("VGG19_PATIENT"))
    img_height = int(os.getenv("VGG19_IMG_HEIGHT"))
    img_width = int(os.getenv("VGG19_IMG_WIDTH"))
    img_deep = int(os.getenv("VGG19_IMG_DEEP"))
    
    # Cargamos los datos para VGG19
    train_data, validation_data = load_data(DATASET_PATH, (img_height, img_width), batch_size, COLOR_MODE, CLASS_MODE)
    
    # Obtenemos las etiquetas y el número de categorias
    labels = train_data.class_indices
    num_categories = labels.__len__() - 1
    
    # Generamos el modelo VGG19
    vgg19 = create_vgg19(img_height, img_width, img_deep, num_categories, ACTIVATION)
    
    # Generamos los callbacks para VGG19
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=RESTORE_BEST_WEIGHTS)
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks=[early_stopping, checkpoint]
 
    # Entrenamos el modelo VGG19
    history = train(
        model = vgg19,
        train_data = train_data,
        validation_data = validation_data,
        epochs = epochs,
        callbacks = callbacks
    )
    
    # Guardamos el modelo VGG19
    acc = round(history.history["accuracy"][-1] * 100)
    vgg19.save(f'{model_path}_{acc}.h5', save_format='tf')