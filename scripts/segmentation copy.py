import os
import tensorflow as tf
import matplotlib as plt
from matplotlib import pyplot
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import cv2

from dotenv import find_dotenv, load_dotenv

from classes.dataset import Dataset
from classes.models.unet import UNet

from sklearn.model_selection import train_test_split

def format_path(path) -> str:
    return path.replace("/", os.sep).replace("\\", os.sep)

def create_df(data_dir):
    images_paths = []
    masks_paths = glob(f'{data_dir}/*/*_mask*')

    for i in masks_paths:
        images_paths.append(i.replace('_mask', ''))

    df = pd.DataFrame(data= {'images_paths': images_paths, 'masks_paths': masks_paths})

    return df

# Function to split dataframe into train, valid, test
def split_df(df):
    # create train_df
    train_df, dummy_df = train_test_split(df, train_size= 0.8)

    # create valid_df and test_df
    valid_df, test_df = train_test_split(dummy_df, train_size= 0.5)

    return train_df, valid_df, test_df

def create_gens(df, aug_dict):
    img_size = (256, 256)
    batch_size = 40

    img_gen = tf.keras.preprocessing.image.ImageDataGenerator(**aug_dict)
    msk_gen = tf.keras.preprocessing.image.ImageDataGenerator(**aug_dict)

    # Create general generator
    image_gen = img_gen.flow_from_dataframe(df, x_col='images_paths', class_mode=None, color_mode='rgb', target_size=img_size,
                                            batch_size=batch_size, save_to_dir=None, save_prefix='image', seed=1)

    mask_gen = msk_gen.flow_from_dataframe(df, x_col='masks_paths', class_mode=None, color_mode='grayscale', target_size=img_size,
                                            batch_size=batch_size, save_to_dir=None, save_prefix= 'mask', seed=1)

    gen = zip(image_gen, mask_gen)

    for (img, msk) in gen:
        img = img / 255
        msk = msk / 255
        msk[msk > 0.5] = 1
        msk[msk <= 0.5] = 0

        yield (img, msk)

# function to create dice coefficient
def dice_coef(y_true, y_pred, smooth=100):
    y_true_flatten = tf.keras.backend.flatten(y_true)
    y_pred_flatten = tf.keras.backend.flatten(y_pred)

    intersection = tf.keras.backend.sum(y_true_flatten * y_pred_flatten)
    union = tf.keras.backend.sum(y_true_flatten) + tf.keras.backend.sum(y_pred_flatten)
    return (2 * intersection + smooth) / (union + smooth)

# function to create dice loss
def dice_loss(y_true, y_pred, smooth=100):
    return -dice_coef(y_true, y_pred, smooth)

# function to create iou coefficient
def iou_coef(y_true, y_pred, smooth=100):
    intersection = tf.keras.backend.sum(y_true * y_pred)
    sum = tf.keras.backend.sum(y_true + y_pred)
    iou = (intersection + smooth) / (sum - intersection + smooth)
    return iou

def show_images(images, masks):
    plt.figure(figsize=(12, 12))
    for i in range(25):
        plt.subplot(5, 5, i+1)
        img_path = images[i]
        mask_path = masks[i]
        # read image and convert it to RGB scale
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # read mask
        mask = cv2.imread(mask_path)
        # sho image and mask
        plt.imshow(image)
        plt.imshow(mask, alpha=0.4)

        plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    
    # Cargamos las variables de entorno
    load_dotenv(find_dotenv())
    
    # Comprobamos si se está usando la GPU
    print(f"Dispositivo de entrenamiento: {tf.config.list_physical_devices('GPU')}")
    
    # Establecemos el tamaño de las imágenes
    IMG_WIDTH = int(os.getenv("SEGMENTATION_IMG_WIDTH"))
    IMG_HEIGHT = int(os.getenv("SEGMENTATION_IMG_HEIGHT"))
    IMG_DEEP = int(os.getenv("SEGMENTATION_IMG_DEEP"))
    BATCH_SIZE = int(os.getenv("SEGMENTATION_BATCH_SIZE"))
    
    # Establecemos las constantes de las rutas
    PROJECT_PATH = os.getenv("PROJECT_PATH")
    DATASET_PATH = os.path.join(PROJECT_PATH, os.getenv("DATASET_SEGMENTATION_PATH"))
    
    print(f"Ruta de entreno: {DATASET_PATH}")
    
    dataset = Dataset(DATASET_PATH)
    dataset.configure(IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE, "rgb", 'categorical')
    dataset.apply_data_augmentation()
    dataset.set_data()
    
    # Mostramos las categorias
    print(f"Las categorias son: {dataset.labels}")
    print(f"Numero de categorias: {dataset.num_categories}")
    
    # Creamos las constantes para guardar el modelo
    MODEL_PATH = os.path.join(PROJECT_PATH, os.getenv("SEGMENTATION_MODEL_PATH"))
    MODEL_NAME = os.path.join(MODEL_PATH, "UNet", "UNet")
    
    # Establecemos los epcohs y el patient
    EPOCHS = int(os.getenv("SEGMENTATION_EPOCHS"))
    PATIENT = int(os.getenv("SEGMENTATION_PATIENT"))
    
    unet = UNet(MODEL_NAME)
    
    unet.create_model(IMG_HEIGHT, IMG_WIDTH, IMG_DEEP, dataset.num_categories)
    unet.compile(tf.optimizers.Adam(1e4), tf.metrics.Accuracy(), 'categorical_crossentropy')
    unet.set_early_stopping(PATIENT)
    unet.set_checkpoint()
    
    unet.train(dataset.train_data, EPOCHS, dataset.validation_data)
    
    # Evaluamos el modelo
    train_loss, train_success = unet.evaluate(dataset.train_data)
    validation_loss, validation_success = unet.evaluate(dataset.validation_data)
    
    unet.save(MODEL_NAME)