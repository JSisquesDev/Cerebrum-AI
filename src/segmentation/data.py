from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd

def create_dataframe():
    # Create data frame and split data on train set, validation set and test set
    df = pd.DataFrame(data={"filename": train_files, 'mask' : mask_files})
    df_train, df_test = train_test_split(df,test_size = 0.1)
    df_train, df_val = train_test_split(df_train,test_size = 0.2)
    print(df_train.values.shape)
    print(df_val.values.shape)
    print(df_test.values.shape)

def preprocess_data():
    image_datagen = ImageDataGenerator(
        rotation_range=0.2,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.05,
        zoom_range=0.05,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    mask_datagen = ImageDataGenerator(
        rotation_range=0.2,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.05,
        zoom_range=0.05,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    return (image_datagen, mask_datagen)

def load_data(data_path, target_size, batch_size, color_mode, class_mode):
    image_datagen, mask_datagen = preprocess_data()
    
    train_data = image_datagen.flow_from_dataframe(
        data_frame,
        x_col = "filename",
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed
    )
        
    validation_data = mask_datagen.flow_from_dataframe()
        data_frame,
        x_col = "mask",
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed
    )
    
    return (train_data, validation_data)