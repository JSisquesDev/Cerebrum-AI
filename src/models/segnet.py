import tensorflow.keras as kr
import tensorflow as tf

# Constantes
SMOOTH = 100

def dice_coef(y_true, y_pred):
    y_truef=tf.keras.backend.flatten(y_true)
    y_predf=tf.keras.backend.flatten(y_pred)
    And=tf.keras.backend.sum(y_truef* y_predf)
    return((2* And + SMOOTH) / (tf.keras.backend.sum(y_truef) + tf.keras.backend.sum(y_predf) + SMOOTH))

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def iou(y_true, y_pred):
    intersection = tf.keras.backend.sum(y_true * y_pred)
    sum_ = tf.keras.backend.sum(y_true + y_pred)
    jac = (intersection + SMOOTH) / (sum_ - intersection + SMOOTH)
    return jac

def jac_distance(y_true, y_pred):
    y_truef=tf.keras.backend.flatten(y_true)
    y_predf=tf.keras.backend.flatten(y_pred)

    return - iou(y_true, y_pred)

def create_model(img_height, img_width, img_deep, activation, epochs):
    # Capa de entrada
    inputs = tf.keras.layers.Input((img_height, img_width, img_deep))
    
    # Encoder
    # Bloque 01
    conv1 = tf.keras.layers.Conv2D(64, (3, 3), padding='same')(inputs)
    act1 = tf.keras.layers.Activation('relu')(conv1)
    conv1 = tf.keras.layers.Conv2D(64, (3, 3), padding='same')(act1)
    bn1 = tf.keras.layers.BatchNormalization(axis=3)(conv1)
    act1 = tf.keras.layers.Activation('relu')(bn1)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(act1)

    # Bloque 02
    conv2 = tf.keras.layers.Conv2D(128, (3, 3), padding='same')(pool1)
    act2 = tf.keras.layers.Activation('relu')(conv2)
    conv2 = tf.keras.layers.Conv2D(128, (3, 3), padding='same')(act2)
    bn2 = tf.keras.layers.BatchNormalization(axis=3)(conv2)
    act2 = tf.keras.layers.Activation('relu')(bn2)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(act2)

    # Bloque 03
    conv3 = tf.keras.layers.Conv2D(256, (3, 3), padding='same')(pool2)
    act3 = tf.keras.layers.Activation('relu')(conv3)
    conv3 = tf.keras.layers.Conv2D(256, (3, 3), padding='same')(act3)
    bn3 = tf.keras.layers.BatchNormalization(axis=3)(conv3)
    act3 = tf.keras.layers.Activation('relu')(bn3)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(act3)

    # Bloque 02
    conv4 = tf.keras.layers.Conv2D(512, (3, 3), padding='same')(pool3)
    act4 = tf.keras.layers.Activation('relu')(conv4)
    conv4 = tf.keras.layers.Conv2D(512, (3, 3), padding='same')(act4)
    bn4 = tf.keras.layers.BatchNormalization(axis=3)(conv4)
    act4 = tf.keras.layers.Activation('relu')(bn4)
    pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(act4)

    # Bloque 05
    conv5 = tf.keras.layers.Conv2D(512, (3, 3), padding='same')(pool4)
    act5 = tf.keras.layers.Activation('relu')(conv5)
    conv5 = tf.keras.layers.Conv2D(512, (3, 3), padding='same')(act5)
    bn5 = tf.keras.layers.BatchNormalization(axis=3)(conv5)
    act5 = tf.keras.layers.Activation('relu')(bn5)
    pool5 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(act5)

    # Decoder
    # Bloque 06
    up6 = tf.keras.layers.UpSampling2D(size=(2, 2))(pool5)
    conv6 = tf.keras.layers.Conv2D(512, (3, 3), padding='same')(up6)
    act6 = tf.keras.layers.Activation('relu')(conv6)
    conv6 = tf.keras.layers.Conv2D(512, (3, 3), padding='same')(act6)
    bn6 = tf.keras.layers.BatchNormalization(axis=3)(conv6)
    act6 = tf.keras.layers.Activation('relu')(bn6)

    # Bloque 07
    up7 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv6)
    conv7 = tf.keras.layers.Conv2D(512, (3, 3), padding='same')(up7)
    act7 = tf.keras.layers.Activation('relu')(conv7)
    conv7 = tf.keras.layers.Conv2D(512, (3, 3), padding='same')(act7)
    bn7 = tf.keras.layers.BatchNormalization(axis=3)(conv7)
    act7 = tf.keras.layers.Activation('relu')(bn7)

    # Bloque 08
    up8 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv7)
    conv8 = tf.keras.layers.Conv2D(256, (3, 3), padding='same')(up8)
    act8 = tf.keras.layers.Activation('relu')(conv8)
    conv8 = tf.keras.layers.Conv2D(256, (3, 3), padding='same')(act8)
    bn8 = tf.keras.layers.BatchNormalization(axis=3)(conv8)
    act8 = tf.keras.layers.Activation('relu')(bn8)

    # Bloque 09
    up9 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv8)
    conv9 = tf.keras.layers.Conv2D(128, (3, 3), padding='same')(up9)
    act9 = tf.keras.layers.Activation('relu')(conv9)
    conv9 = tf.keras.layers.Conv2D(128, (3, 3), padding='same')(act9)
    bn9 = tf.keras.layers.BatchNormalization(axis=3)(conv9)
    act9 = tf.keras.layers.Activation('relu')(bn9)

    # Bloque 10
    up10 = tf.keras.layers.UpSampling2D(size=(2, 2))(conv9)
    conv10 = tf.keras.layers.Conv2D(64, (3, 3), padding='same')(up10)
    act10 = tf.keras.layers.Activation('relu')(conv10)
    conv10 = tf.keras.layers.Conv2D(64, (3, 3), padding='same')(act10)
    bn10 = tf.keras.layers.BatchNormalization(axis=3)(conv10)
    act10 = tf.keras.layers.Activation('relu')(bn10)

    # Capa de salida
    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation=activation)(act10)

    model = tf.keras.models.Model(inputs=[inputs], outputs=[outputs])
    
    # Establecemos el learning rate
    learning_rate = 1e-4
    
    decay_rate = learning_rate / epochs
    optimizer = tf.keras.optimizers.Adamax(learning_rate=learning_rate)
    
    model.compile(
        optimizer = optimizer,
        metrics=["binary_accuracy", iou, dice_coef],
        loss = dice_coef_loss
    )
    
    model.summary()
    
    return model