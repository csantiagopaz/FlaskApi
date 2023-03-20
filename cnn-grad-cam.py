import numpy as np
import cv2
import tensorflow as tf
import os
from PIL import Image
from tensorflow.keras.applications.vgg16 import VGG16 as Model
from tensorflow.keras.applications import vgg16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img
from tf_keras_vis.utils import normalize
from matplotlib import cm
from tf_keras_vis.gradcam import Gradcam
from load_data import load_pipeline

#predict a class using vgg16 model
def predict(model, pipe):
    predictions = []
    for img in pipe:
        predict_img = model.predict(img)
        predictions.append(vgg16.decode_predictions(predict_img, top=1))
    return predictions
# define the loss functions with a target clas

def score(output):

    return output[0][np.argmax(output[0])]

def model_modifier(md1):
    md1.layers[-1].activation = tf.keras.activations.linear # whe change the activation of the last layer to linear

# define the grad cam function
def generate_tensor(data_dir='data', size=224):
    for img_data in os.listdir(data_dir):
        img_data = os.path.join(data_dir, img_data)
        img = load_img(img_data, target_size=(size,size))
        img1_array = cv2.cvtColor(np.array(img),cv2.COLOR_RGB2BGR)
        x = np.asarray([np.array(img)])
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        tensor_x = tf.convert_to_tensor(x, dtype=tf.float32)
        yield tensor_x,img1_array
# Create a TensorFlow tensor from the NumPy array

def generate_gradcam(model, model_modifier, results):
    i= 0
    for output, tensor in zip(results, generate_tensor()):
        i+=1
        print(output, tensor)
        gradcam = Gradcam(model, model_modifier=model_modifier, clone=False)
        cam = gradcam(score, tensor[0][0], penultimate_layer=-1)
        cam = normalize(cam)
        original_img_array = tensor[1]
        heatmapImg = np.uint8(cm.jet(cam[0])[..., :3]* 255)
        # # change the color map to jet
        heatmapImg = cv2.applyColorMap(heatmapImg, cv2.COLORMAP_JET)
        # # lets add some alpha transparency
        alpha = 0.5
        overlay_img = heatmapImg.copy()
        print(type(original_img_array))
        result1 = cv2.addWeighted(original_img_array, alpha, heatmapImg, 1-alpha, 0)
        scale_percent = 200
        w = int(heatmapImg.shape[1]* scale_percent/100)
        h = int(heatmapImg.shape[1]* scale_percent/100)
        dim = (w,h)

        result1 = cv2.resize(result1, dim, interpolation=cv2.INTER_AREA)
        original_img_array = cv2.resize(original_img_array, dim, interpolation=cv2.INTER_AREA)

        original_img = Image.fromarray(original_img_array)
        gradcam_img = Image.fromarray(result1)

        gradcam_img.save(f"grad_cam_{i}.jpeg")
    with open('predictions.txt', 'a+', encoding='utf-8') as file:
        file.write('\n'.join([result for result in results]))


# load the Vgg16 model
gpus= tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

model = Model(weights='imagenet', include_top=True)




results = predict(model, load_pipeline('data', 224))
generate_gradcam(model, model_modifier=model_modifier, results=results)
