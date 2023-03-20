# Correr contenedor de Tensorflow

sudo docker run --dns 8.8.8.8 --gpus all -v $PWD:/test -it tensorflow/tensorflow:latest-gpu

# Correr contenedor con Pytorch

sudo docker run -it -v $PWD:/test --gpus all pytorch/pytorch

# opencv dependecies not found original in docker img
error: {
    ImportError: libGL.so.1: cannot open shared object file: No such file or directory}

apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
