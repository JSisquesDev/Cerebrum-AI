import tensorflow.keras as kr
import tensorflow as tf

def create_model(img_height, img_width, img_deep, num_categories, activation):
    model  = kr.Sequential([
        # Bloque 01
        kr.layers.Conv2D(64, activation='relu', kernel_size=(3, 3), input_shape=(img_height, img_width, img_deep)),
        kr.layers.Conv2D(64, activation='relu', kernel_size=(3, 3), padding='same'),
        kr.layers.MaxPooling2D(2,2),

        # Bloque 02
        kr.layers.Conv2D(128, activation='relu', kernel_size=(3, 3), padding='same'),
        kr.layers.Conv2D(128, activation='relu', kernel_size=(3, 3), padding='same'),
        kr.layers.MaxPooling2D(2,2),

        # Bloque 03
        kr.layers.Conv2D(256, activation='relu', kernel_size=(3, 3), padding='same'),
        kr.layers.Conv2D(256, activation='relu', kernel_size=(3, 3), padding='same'),
        kr.layers.Conv2D(256, activation='relu', kernel_size=(3, 3), padding='same'),
        kr.layers.Conv2D(256, activation='relu', kernel_size=(3, 3), padding='same'),
        kr.layers.MaxPooling2D(2,2),
        
        # Bloque 04
        kr.layers.Conv2D(512, activation='relu', kernel_size=(3, 3), padding='same'),
        kr.layers.Conv2D(512, activation='relu', kernel_size=(3, 3), padding='same'),
        kr.layers.Conv2D(512, activation='relu', kernel_size=(3, 3), padding='same'),
        kr.layers.Conv2D(512, activation='relu', kernel_size=(3, 3), padding='same'),
        kr.layers.MaxPooling2D(2,2),
        
        # Bloque 05
        kr.layers.Conv2D(512, activation='relu', kernel_size=(3, 3), padding='same'),
        kr.layers.Conv2D(512, activation='relu', kernel_size=(3, 3), padding='same'),
        kr.layers.Conv2D(512, activation='relu', kernel_size=(3, 3), padding='same'),
        kr.layers.Conv2D(512, activation='relu', kernel_size=(3, 3), padding='same'),
        kr.layers.MaxPooling2D(2,2),

        # Bloque 06
        kr.layers.Flatten(),

        # Bloque 07
        kr.layers.Dense(4096, activation='relu'),
        kr.layers.Dropout(0.3),
        kr.layers.Dense(4096, activation='relu'),
        kr.layers.Dropout(0.3),
        kr.layers.Dense(num_categories, activation=activation)
    ])
    
    model.compile(
        optimizer = tf.optimizers.Adam(learning_rate=1e-4), 
        metrics = ['accuracy'],
        loss = 'binary_crossentropy'
        )
    
    model.summary()
    
    return model