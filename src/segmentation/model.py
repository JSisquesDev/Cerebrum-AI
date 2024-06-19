from src.models import unet

def create_unet(img_height, img_width, img_deep, activation, epochs):
    return unet.create_model(
        img_height = img_height,
        img_width = img_width,
        img_deep = img_deep,
        activation = activation,
        epochs=epochs
    )