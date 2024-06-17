from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from dotenv import load_dotenv
from model import create_alexnet, create_vgg19
from data import load_train_data, load_validation_data 

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
    print(f"Tiempo total de entrenamiento para AlexNet: {train_time}")
    
    return result

if __name__ == '__main__':
    # Cargamos las variables de entorno para la detección
    ENV = load_dotenv(f'.{os.sep}config{os.sep}.env.detection')
    
    # Establecemos las variables comunes a los modelos
    OUTPUT_PATH = str(os.getenv("MODEL_PATH"))
    RESTORE_BEST_WEIGHTS = bool(os.getenv("RESTORE_BEST_WEIGHTS"))
    ACTIVATION = str(os.getenv('ACTIVATION'))
    
    # Cargamos las variables para el modelo AlexNet
    model_name = str(os.getenv('ALEXNET_MODEL_NAME'))
    epochs = int(os.getenv("ALEXNET_EPOCHS"))
    patience = int(os.getenv("ALEXNET_PATIENT"))
    img_height = int(os.getenv("ALEXNET_IMG_HEIGHT"))
    img_width = int(os.getenv("ALEXNET_IMG_WIDTH"))
    img_deep = int(os.getenv("ALEXNET_IMG_DEEP"))
    num_categories = 1
    
    # Cargamos los datos para AlexNet
    train_data = load_train_data()
    validation_data = load_validation_data()
    
    # Generamos el modelo Alexnet
    alexnet = create_alexnet(img_height, img_width, img_deep, num_categories, ACTIVATION)
    
    # Generamos los callbacks para Alexnet
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=RESTORE_BEST_WEIGHTS)
    checkpoint = ModelCheckpoint(f"{OUTPUT_PATH}{os.sep}{model_name}.h5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks=[early_stopping, checkpoint]
 
    # Entrenamos el modelo AlexNet
    train(
        model = alexnet,
        train_data = train_data,
        validation_data = validation_data,
        epochs = epochs,
        callbacks = callbacks
    )
    
    # Hacer lo mismo para VGG19