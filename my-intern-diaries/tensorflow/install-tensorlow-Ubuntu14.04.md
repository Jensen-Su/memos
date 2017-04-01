1. Install `anaconda`

    $ wget http://repo.continuum.io/archive/Anaconda2-4.1.1-Linux_x86_64.sh
    $ bash Anaconda2-4.1.1-Linux_x86_64.sh

Restart your terminal.

2. Install tensorflow:

    $ conda create -n tensorflow python=2.7
    $ source activate tensorflow
    $ pip install --ignore-install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.0.1-cp27-none-linux_x86_64.whl
    $ export TF_BINARY_URL=https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.0.1-cp27-none-linux_x86_64.whl
    $ pip install --ignore-install $TF_BINARY_URL 

3. install keras:

    $ sudo pip install keras
    $ mkdir ~/.keras
    $ vim ~/.keras/keras.jason
        {
            "image_data_format": "channels_last",
            "epsilon": 1e-07,
            "floatx": "float32",
            "backend": "tensorflow"
        }
    $ KERAS_BACKEND=tensorflow python -c "from keras import backend"

4. Install opencv if necessary:

    $ conda install -c menpo opencv3=3.2.0
    
    
