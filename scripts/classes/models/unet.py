from .. import model

import tensorflow.keras as kr

class UNet(model.Model):
    def __init__(self, name) -> None:
        self.name = name
    
    def create_model(self, img_height, img_width, img_deep, num_categories, activation='sigmoid'):
        
        inputs = kr.layers.Input((img_height, img_width, img_deep))

        # First DownConvolution / Encoder Leg will begin, so start with Conv2D
        conv1 = kr.layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same")(inputs)
        bn1 = kr.layers.Activation("relu")(conv1)
        conv1 = kr.layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same")(bn1)
        bn1 = kr.layers.BatchNormalization(axis=3)(conv1)
        bn1 = kr.layers.Activation("relu")(bn1)
        pool1 = kr.layers.MaxPooling2D(pool_size=(2, 2))(bn1)

        conv2 = kr.layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same")(pool1)
        bn2 = kr.layers.Activation("relu")(conv2)
        conv2 = kr.layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same")(bn2)
        bn2 = kr.layers.BatchNormalization(axis=3)(conv2)
        bn2 = kr.layers.Activation("relu")(bn2)
        pool2 = kr.layers.MaxPooling2D(pool_size=(2, 2))(bn2)

        conv3 = kr.layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same")(pool2)
        bn3 = kr.layers.Activation("relu")(conv3)
        conv3 = kr.layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same")(bn3)
        bn3 = kr.layers.BatchNormalization(axis=3)(conv3)
        bn3 = kr.layers.Activation("relu")(bn3)
        pool3 = kr.layers.MaxPooling2D(pool_size=(2, 2))(bn3)

        conv4 = kr.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same")(pool3)
        bn4 = kr.layers.Activation("relu")(conv4)
        conv4 = kr.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same")(bn4)
        bn4 = kr.layers.BatchNormalization(axis=3)(conv4)
        bn4 = kr.layers.Activation("relu")(bn4)
        pool4 = kr.layers.MaxPooling2D(pool_size=(2, 2))(bn4)

        conv5 = kr.layers.Conv2D(filters=1024, kernel_size=(3, 3), padding="same")(pool4)
        bn5 = kr.layers.Activation("relu")(conv5)
        conv5 = kr.layers.Conv2D(filters=1024, kernel_size=(3, 3), padding="same")(bn5)
        bn5 = kr.layers.BatchNormalization(axis=3)(conv5)
        bn5 = kr.layers.Activation("relu")(bn5)

        """ Now UpConvolution / Decoder Leg will begin, so start with Conv2DTranspose
        The gray arrows (in the above image) indicate the skip connections that concatenate the encoder feature map with the decoder, which helps the backward flow of gradients for improved training. """
        """ After every concatenation we again apply two consecutive regular convolutions so that the model can learn to assemble a more precise output """

        up6 = kr.layers.concatenate([kr.layers.Conv2DTranspose(512, kernel_size=(2, 2), strides=(2, 2), padding="same")(bn5), conv4], axis=3)
        conv6 = kr.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same")(up6)
        bn6 = kr.layers.Activation("relu")(conv6)
        conv6 = kr.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same")(bn6)
        bn6 = kr.layers.BatchNormalization(axis=3)(conv6)
        bn6 = kr.layers.Activation("relu")(bn6)

        up7 = kr.layers.concatenate([kr.layers.Conv2DTranspose(256, kernel_size=(2, 2), strides=(2, 2), padding="same")(bn6), conv3], axis=3)
        conv7 = kr.layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same")(up7)
        bn7 = kr.layers.Activation("relu")(conv7)
        conv7 = kr.layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same")(bn7)
        bn7 = kr.layers.BatchNormalization(axis=3)(conv7)
        bn7 = kr.layers.Activation("relu")(bn7)

        up8 = kr.layers.concatenate([kr.layers.Conv2DTranspose(128, kernel_size=(2, 2), strides=(2, 2), padding="same")(bn7), conv2], axis=3)
        conv8 = kr.layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same")(up8)
        bn8 = kr.layers.Activation("relu")(conv8)
        conv8 = kr.layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same")(bn8)
        bn8 = kr.layers.BatchNormalization(axis=3)(conv8)
        bn8 = kr.layers.Activation("relu")(bn8)

        up9 = kr.layers.concatenate([kr.layers.Conv2DTranspose(64, kernel_size=(2, 2), strides=(2, 2), padding="same")(bn8), conv1], axis=3)
        conv9 = kr.layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same")(up9)
        bn9 = kr.layers.Activation("relu")(conv9)
        conv9 = kr.layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same")(bn9)
        bn9 = kr.layers.BatchNormalization(axis=3)(conv9)
        bn9 = kr.layers.Activation("relu")(bn9)

        conv10 = kr.layers.Conv2D(filters=num_categories, kernel_size=(1, 1), activation=activation)(bn9)

        self.model = kr.Model(inputs=[inputs], outputs=[conv10])