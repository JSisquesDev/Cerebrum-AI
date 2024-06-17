from models import alexnet
from models import vgg19

def create_alexnet(img_height, img_width, img_deep, num_categories, activation):
    return alexnet.create_model(
        img_height = img_height,
        img_width = img_width,
        img_deep = img_deep,
        num_categories = num_categories,
        activation = activation
    )

def create_vgg19(img_height, img_width, img_deep, num_categories, activation):
    return vgg19.create_model(
        img_height = img_height,
        img_width = img_width,
        img_deep = img_deep,
        num_categories = num_categories,
        activation = activation
    )