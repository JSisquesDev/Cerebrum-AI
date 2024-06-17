from tensorflow.keras.preprocessing.image import ImageDataGenerator

def preprocess_data():
    return ImageDataGenerator(
        rescale = 1.0/255,
        horizontal_flip = True,
        vertical_flip = True,
        validation_split = 0.2,
        rotation_range = 20
        )

def load_data(data_path, target_size, batch_size, color_mode, class_mode):
    image_data_generator = preprocess_data()
    
    train_data = image_data_generator.flow_from_directory(
            data_path,
            target_size = target_size,
            batch_size = batch_size,
            color_mode = color_mode,
            shuffle = True,
            class_mode = class_mode,
            subset = 'training',
        )
        
    validation_data = image_data_generator.flow_from_directory(
        data_path,
        target_size = target_size,
        batch_size = batch_size,
        color_mode = color_mode,
        shuffle = False,
        class_mode = class_mode,
        subset = 'validation',
    )
    
    return (train_data, validation_data)