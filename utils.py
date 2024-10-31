from keras._tf_keras.keras.preprocessing import image
import numpy as np
import io, json

IMAGE_SIZE = (224, 224)

def load_class_names(classes_path):
    """
    Loads class names from a JSON file.

    :classes_path: File path of the JSON file containing class names.

    :returns: list of class names.
    """
    try:
        with open(classes_path, 'r') as f:
            class_names = json.load(f)
        return class_names
    except Exception as e:
        print(f"Error loading class names from {classes_path}: {e}")
        return []

def prepare_image(img):
    """
    Prepares an image for model prediction by loading, resizing, and normalizing it.
    
    :img: Path to the image file.
    """
    img = image.load_img(io.BytesIO(img.read()), target_size=IMAGE_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0 
    return img_array