from src.models import unet

def create_unet(img_height, img_width, img_deep, num_categories, activation):
    return unet.create_model(
        img_height = img_height,
        img_width = img_width,
        img_deep = img_deep,
        num_categories = num_categories,
        activation = activation
    )