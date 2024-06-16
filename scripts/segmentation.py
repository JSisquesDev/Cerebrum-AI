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

from glob import glob

def format_path(path) -> str:
    return path.replace("/", os.sep).replace("\\", os.sep)

def train_generator(data_frame, batch_size, aug_dict,
        image_color_mode="rgb",
        mask_color_mode="grayscale",
        image_save_prefix="image",
        mask_save_prefix="mask",
        save_to_dir=None,
        target_size=(256,256),
        seed=1):
    '''
    can generate image and mask at the same time use the same seed for
    image_datagen and mask_datagen to ensure the transformation for image
    and mask is the same if you want to visualize the results of generator,
    set save_to_dir = "your path"
    '''
    image_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**aug_dict)
    mask_datagen = tf.keras.preprocessing.image.ImageDataGenerator(**aug_dict)
    
    image_generator = image_datagen.flow_from_dataframe(
        data_frame,
        x_col = "filename",
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)

    mask_generator = mask_datagen.flow_from_dataframe(
        data_frame,
        x_col = "mask",
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)

    train_gen = zip(image_generator, mask_generator)
    
    for (img, mask) in train_gen:
        img, mask = adjust_data(img, mask)
        yield (img,mask)

def adjust_data(img,mask):
    img = img / 255
    mask = mask / 255
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0
    
    return (img, mask)

smooth=100

def dice_coef(y_true, y_pred):
    y_truef=tf.keras.backend.flatten(y_true)
    y_predf=tf.keras.backend.flatten(y_pred)
    And=tf.keras.backend.sum(y_truef* y_predf)
    return((2* And + smooth) / (tf.keras.backend.sum(y_truef) + tf.keras.backend.sum(y_predf) + smooth))

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def iou(y_true, y_pred):
    intersection = tf.keras.backend.sum(y_true * y_pred)
    sum_ = tf.keras.backend.sum(y_true + y_pred)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac

def jac_distance(y_true, y_pred):
    y_truef=tf.keras.backend.flatten(y_true)
    y_predf=tf.keras.backend.flatten(y_pred)

    return - iou(y_true, y_pred)

def unet(input_size=(256,256,3)):
    inputs = tf.keras.layers.Input(input_size)
    
    conv1 = tf.keras.layers.Conv2D(64, (3, 3), padding='same')(inputs)
    bn1 = tf.keras.layers.Activation('relu')(conv1)
    conv1 = tf.keras.layers.Conv2D(64, (3, 3), padding='same')(bn1)
    bn1 = tf.keras.layers.BatchNormalization(axis=3)(conv1)
    bn1 = tf.keras.layers.Activation('relu')(bn1)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(bn1)

    conv2 = tf.keras.layers.Conv2D(128, (3, 3), padding='same')(pool1)
    bn2 = tf.keras.layers.Activation('relu')(conv2)
    conv2 = tf.keras.layers.Conv2D(128, (3, 3), padding='same')(bn2)
    bn2 = tf.keras.layers.BatchNormalization(axis=3)(conv2)
    bn2 = tf.keras.layers.Activation('relu')(bn2)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(bn2)

    conv3 = tf.keras.layers.Conv2D(256, (3, 3), padding='same')(pool2)
    bn3 = tf.keras.layers.Activation('relu')(conv3)
    conv3 = tf.keras.layers.Conv2D(256, (3, 3), padding='same')(bn3)
    bn3 = tf.keras.layers.BatchNormalization(axis=3)(conv3)
    bn3 = tf.keras.layers.Activation('relu')(bn3)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(bn3)

    conv4 = tf.keras.layers.Conv2D(512, (3, 3), padding='same')(pool3)
    bn4 = tf.keras.layers.Activation('relu')(conv4)
    conv4 = tf.keras.layers.Conv2D(512, (3, 3), padding='same')(bn4)
    bn4 = tf.keras.layers.BatchNormalization(axis=3)(conv4)
    bn4 = tf.keras.layers.Activation('relu')(bn4)
    pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(bn4)

    conv5 = tf.keras.layers.Conv2D(1024, (3, 3), padding='same')(pool4)
    bn5 = tf.keras.layers.Activation('relu')(conv5)
    conv5 = tf.keras.layers.Conv2D(1024, (3, 3), padding='same')(bn5)
    bn5 = tf.keras.layers.BatchNormalization(axis=3)(conv5)
    bn5 = tf.keras.layers.Activation('relu')(bn5)

    up6 = tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(bn5), conv4], axis=3)
    conv6 = tf.keras.layers.Conv2D(512, (3, 3), padding='same')(up6)
    bn6 = tf.keras.layers.Activation('relu')(conv6)
    conv6 = tf.keras.layers.Conv2D(512, (3, 3), padding='same')(bn6)
    bn6 = tf.keras.layers.BatchNormalization(axis=3)(conv6)
    bn6 = tf.keras.layers.Activation('relu')(bn6)

    up7 = tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(bn6), conv3], axis=3)
    conv7 = tf.keras.layers.Conv2D(256, (3, 3), padding='same')(up7)
    bn7 = tf.keras.layers.Activation('relu')(conv7)
    conv7 = tf.keras.layers.Conv2D(256, (3, 3), padding='same')(bn7)
    bn7 = tf.keras.layers.BatchNormalization(axis=3)(conv7)
    bn7 = tf.keras.layers.Activation('relu')(bn7)

    up8 = tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(bn7), conv2], axis=3)
    conv8 = tf.keras.layers.Conv2D(128, (3, 3), padding='same')(up8)
    bn8 = tf.keras.layers.Activation('relu')(conv8)
    conv8 = tf.keras.layers.Conv2D(128, (3, 3), padding='same')(bn8)
    bn8 = tf.keras.layers.BatchNormalization(axis=3)(conv8)
    bn8 = tf.keras.layers.Activation('relu')(bn8)

    up9 = tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(bn8), conv1], axis=3)
    conv9 = tf.keras.layers.Conv2D(64, (3, 3), padding='same')(up9)
    bn9 = tf.keras.layers.Activation('relu')(conv9)
    conv9 = tf.keras.layers.Conv2D(64, (3, 3), padding='same')(bn9)
    bn9 = tf.keras.layers.BatchNormalization(axis=3)(conv9)
    bn9 = tf.keras.layers.Activation('relu')(bn9)

    conv10 = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(bn9)

    return tf.keras.models.Model(inputs=[inputs], outputs=[conv10])

if __name__ == '__main__':
    
    # Cargamos las variables de entorno
    load_dotenv(find_dotenv())
    
    os.environ["TF_DIRECTML_MAX_ALLOC_SIZE"] = "536870912" # 512MB
    os.environ["TF_CPP_MIN_VLOG_LEVEL"] = "3"
    
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
    
    ##Archivos
    train_files = []
    mask_files = glob('../datasets/segmentation/kaggle_3m/*/*_mask*')

    for i in mask_files:
        train_files.append(i.replace('_mask',''))

    print(train_files[:10])
    print(mask_files[:10])
    
    ## Data visualization
    rows,cols=3,3
    fig=plt.figure(figsize=(10,10))
    for i in range(1,rows*cols+1):
        fig.add_subplot(rows,cols,i)
        img_path=train_files[i]
        msk_path=mask_files[i]
        img=cv2.imread(img_path)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        msk=cv2.imread(msk_path)
        plt.imshow(img)
        plt.imshow(msk,alpha=0.4)
    plt.show()
    
    ## Create data frame and split data on train set, validation set and test set
    df = pd.DataFrame(data={"filename": train_files, 'mask' : mask_files})
    df_train, df_test = train_test_split(df,test_size = 0.1)
    df_train, df_val = train_test_split(df_train,test_size = 0.2)
    print(df_train.values.shape)
    print(df_val.values.shape)
    print(df_test.values.shape)
    
    ## modelo
    
    model = unet()
    model.summary()
    
    train_generator_args = dict(rotation_range=0.2,
                            width_shift_range=0.05,
                            height_shift_range=0.05,
                            shear_range=0.05,
                            zoom_range=0.05,
                            horizontal_flip=True,
                            fill_mode='nearest')
    train_gen = train_generator(df_train, BATCH_SIZE,
                                    train_generator_args,
                                    target_size=(IMG_HEIGHT, IMG_WIDTH))
        
    test_gener = train_generator(df_val, BATCH_SIZE,
                                    dict(),
                                    target_size=(IMG_HEIGHT, IMG_WIDTH))
        
    model = unet(input_size=(IMG_HEIGHT, IMG_WIDTH, IMG_DEEP))

    learning_rate = 1e-4
    
    EPOCHS = int(os.getenv("SEGMENTATION_EPOCHS"))
    PATIENT = int(os.getenv("SEGMENTATION_PATIENT"))

    decay_rate = learning_rate / EPOCHS
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=decay_rate, amsgrad=False)
    model.compile(optimizer=opt, loss=dice_coef_loss, metrics=["binary_accuracy", iou, dice_coef])

    callbacks = [tf.keras.callbacks.ModelCheckpoint('unet_brain_mri_seg.hdf5', verbose=1, save_best_only=True)]

    history = model.fit(train_gen,
                        steps_per_epoch=len(df_train) / BATCH_SIZE, 
                        epochs=EPOCHS, 
                        callbacks=callbacks,
                        validation_data = test_gener,
                        validation_steps=len(df_val) / BATCH_SIZE)
    
    ##
    
    a = history.history

    list_traindice = a['dice_coef']
    list_testdice = a['val_dice_coef']

    list_trainjaccard = a['iou']
    list_testjaccard = a['val_iou']

    list_trainloss = a['loss']
    list_testloss = a['val_loss']
    plt.figure(1)
    plt.plot(list_testloss, 'b-')
    plt.plot(list_trainloss,'r-')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.title('loss graph', fontsize = 15)
    plt.figure(2)
    plt.plot(list_traindice, 'r-')
    plt.plot(list_testdice, 'b-')
    plt.xlabel('iteration')
    plt.ylabel('accuracy')
    plt.title('accuracy graph', fontsize = 15)
    plt.show()
    
    '''
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
    
    unet.create_model(IMG_HEIGHT, IMG_WIDTH, IMG_DEEP, 1)
    unet.compile(tf.optimizers.Adam(1e4), tf.metrics.Accuracy(), 'categorical_crossentropy')
    unet.set_early_stopping(PATIENT)
    unet.set_checkpoint()
    
    unet.train(dataset.train_data, EPOCHS, dataset.validation_data)
    
    # Evaluamos el modelo
    train_loss, train_success = unet.evaluate(dataset.train_data)
    validation_loss, validation_success = unet.evaluate(dataset.validation_data)
    
    unet.save(MODEL_NAME)
    '''