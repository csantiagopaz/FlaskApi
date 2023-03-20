import numpy as np
import cv2
import os
import imghdr
from PIL import Image
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.applications.vgg16 import preprocess_input
from keras import preprocessing

images = []

def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

data_dir = 'data'
image_exts = ['jpeg', 'jpg', 'bmp', 'png']

def remove_unkown_extensions(data_dir):
    for image in os.listdir(data_dir):
        img_path = os.path.join(data_dir, image)
        try:
            img = cv2.imread(img_path)
            tip = imghdr.what(img_path)
            if tip not in image_exts:
                os.remove(img_path)
                raise FileNotFoundError('The image is not from a kwon extention {map(lambda x: "{x}\n",image_exts)}')
        
        except FileNotFoundError as e:
            print('Issue with image {img_path}')
    return True

def load_data(data_dir):
    data = tf.keras.utils.image_dataset_from_directory(data_dir)
    data_iterator = data.as_numpy_interador()

    batch = data_iterator.next()
    # visualize some data, comment if not necessary
    fig, ax = plt.subplots(ncols=4, figsize=(20,20))
    for idx, img in enumerate(batch[0][:4]):
        ax[idx].imshow(img.astype(int))
        ax[idx].title.set_text(batch[1][idx])

#
def load_pipeline(data_dir, size):
    for img_data in os.listdir(data_dir):
        img_data = os.path.join(data_dir, img_data)
        img = load_img(img_data, target_size=(size,size))
        x = preprocess_input(np.asarray([np.array(img)]))
        tensor_x = tf.convert_to_tensor(x, dtype=tf.float32)
        yield tensor_x

if __name__ == '__main__':
    print(load_pipeline('data', 224))
