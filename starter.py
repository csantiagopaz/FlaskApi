import tensorflow as tf
import os
os.system("pip install --upgrade pip")
os.system("pip install opencv-python pillow matplotlib tf_keras_vis")
os.system("apt-get update && apt-get install ffmpeg libsm6 libxext6  -y")
os.system("apt upgrade")

print(tf.config.list_physical_devices('GPU'))
print('hola mundo')