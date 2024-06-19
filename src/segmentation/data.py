from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from glob import glob

import pandas as pd
import os

def get_training_files(dataset_path):
    # Generamos las listas de los ficheros
    train_files = []
    mask_files = glob(f'{dataset_path}{os.sep}*{os.sep}*_mask*')
    
    # Recorremos los ficheros con marca mask y se le quitamos
    for file in mask_files:
        train_files.append(file.replace('_mask', ''))
        
    print(f'Ruta de los archivos de segmentación: {dataset_path}{os.sep}*{os.sep}*_mask*')
    print(f'Número de archivos sin máscara: {train_files.__len__()}')
    print(f'Número de archivos con máscara: {mask_files.__len__()}')
    
    return (train_files, mask_files)

def adjust_data(img,mask):
    img = img / 255
    mask = mask / 255
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0
    
    return (img, mask)

def split_dataframe(dataframe):

    df_train, df_test = train_test_split(dataframe,test_size = 0.1)
    df_train, df_val = train_test_split(df_train,test_size = 0.2)
    
    print(f'Forma de los datos de entrenamiento: {df_train.values.shape}')
    print(f'Forma de los datos de validación: {df_val.values.shape}')
    print(f'Forma de los datos de test: {df_test.values.shape}')
    
    return (df_train, df_val, df_test) 

def create_dataframe(train_files, mask_files):
    # Create data frame and split data on train set, validation set and test set
    df = pd.DataFrame(data={"filename": train_files, 'mask' : mask_files})
    
    # Verificamos y convertimos los datos a strings
    df['filename'] = df['filename'].astype(str)
    df['mask'] = df['mask'].astype(str)
    
    return df

def train_generator(dataframe, batch_size, aug_dict, image_color_mode, mask_color_mode, image_save_prefix, mask_save_prefix, target_size, seed):
    
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    
    # Establecemos el conjunto de datos de entrenamiento
    image_generator = image_datagen.flow_from_dataframe(
        dataframe,
        x_col = "filename",
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = None,
        save_prefix  = image_save_prefix,
        seed = seed
    )
        
    mask_generator = mask_datagen.flow_from_dataframe(
        dataframe,
        x_col = "mask",
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = None,
        save_prefix  = mask_save_prefix,
        seed = seed
    )
    
    train_gen = zip(image_generator, mask_generator)

    for (img, mask) in train_gen:
        img, mask = adjust_data(img, mask)
        yield (img,mask)

def load_data(dataframe, target_size, batch_size, image_color_mode, image_save_prefix, mask_color_mode, mask_save_prefix, seed):
    
    # Establecemos los argumentos del generador
    generator_args = dict(
        rotation_range=0.2,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.05,
        zoom_range=0.05,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    train_gen = train_generator(
        dataframe = dataframe,
        batch_size = batch_size,
        aug_dict = generator_args,
        image_color_mode = image_color_mode,
        mask_color_mode = mask_color_mode,
        image_save_prefix = image_save_prefix,
        mask_save_prefix = mask_save_prefix,
        target_size = target_size,
        seed = 1
    )
    
    test_gen = train_generator(
        dataframe = dataframe,
        batch_size = batch_size,
        aug_dict = dict(),
        image_color_mode = image_color_mode,
        mask_color_mode = mask_color_mode,
        image_save_prefix = image_save_prefix,
        mask_save_prefix = mask_save_prefix,
        target_size = target_size,
        seed = 1
    )
    
    
    return (train_gen, test_gen)