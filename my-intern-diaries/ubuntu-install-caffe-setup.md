---
Ubuntu Install & Caffe Setup (2016/11/02~2016/11/19)
---

    By Jincheng Su @ Hikvision, mail: jcsu14@fudan.edu.cn

#### 1.1 Install Ubuntu 14.04 (dual disks: SSD + HD)
* Convert a liveCD to a bootable USB.
    Under Windows, one can do it by **writing hardware image into USB disk** in 'UltraISO'.
* Insert the USB disk to your computer and change the 'BIOS' boot setting to boot from USB disk.
* Install Ubuntu -> ... -> choose 'something else'
* Assign disk spaces mounted on root `/` and on `/home` respectively. It is recommended to assign 100 GB for `/` and as large as possible for `/home`. A `swap area` of half of your RAM size is also required.
* Under 'Device for boot loader installation', choose `/dev/sda` which should be already set by default to use Grub to load all systems on this drive `sda`. Click 'Install Now'.
    Note: if you choose `sda#*`, then Ubuntu need to be manually added to driver's boot loader after installation, which is a very tough problem especially when you are installing 'Ubuntu' alongside with a already installed Windows 10 that boot with 'UEFI'.'

* If you cannot boot into Ubuntu, try:

    my '\boot' is mounted to sda5 and '\' to sda6:

        Boot the liveCD (usb ubuntu), try Ubuntu, and open a termial.
        $ sudo mount /dev/sda5 /mnt
        $ sudo grub-install --root-directory=/mnt /dev/sda
        $ sudo reboot

    Note: before reboot, some people would suggest you 'sudo update-grub'.
    This incurred a problem here:
            'error: failed to get cannonical path of /cow',
    which I failed to solve after plenty of efforts.

    Then I discarded this line, and it just worked around.
    The system sucessfully boots, but grub still have some miner problem.

    After `sudo apt-get upgrade`, the grub boot guide just came back.

#### 1.2 Install Nvidia GPU driver

To install the correct driver, firstly purge nvidia*:

    sudo apt-get purge nvidia*; sudo apt-get autoremove

Find out the model of your graphics card:

    $ lspci -vnn | grep -i VGA -A 12

Find out the right driver version for your graphics card:

    curl: http://www.nvidia.com/Download/index.aspx

Setup the xorg-edgers ppa

    $ sudo add-apt-repository ppa:xorg-edgers/ppa -y
    $ apt-get update

Intall the driver

    # the version for this machine is 367.57
    $ sudo apt-get install nvidia-367

NOTE: Never 'sudo apt-get install nvidia-current', which installed a 304 version for me, and caused the problem of `failed to login ubuntu`.

Installing the right GPU driver also solved the 'built-in display' problem if encountered.

#### 1.3 Install MATLAB

#### 1.4 Install CAFFE
* **Install caffe dependencies**

        $ sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libhdf5-serial-dev protobuf-compiler
        $ sud apt-get install --no-install-recommends libboost-all-dev
        $ sudo apt-get install libatlas-base-dev
        $ sudo apt-get install python-dev
        $ sudo apt-get install libgflags-dev libgoogle-glog-dev liblmdb-dev

* **Install Cuda 8.0**

	download .deb from website
		$ sudo dpkg -i cuda-repo-ubuntu1404-8-0-local-8.0.44-1_amd64.deb
		$ sudo apt-get update
		$ sudo apt-get install cuda

    Error info:

         	The following packages have unmet dependencies:
         	unity-control-center : Depends: libcheese-gtk23 (>= 3.4.0) but it is not going to be installed
                                Depends: libcheese7 (>= 3.0.1) but it is not going to be installed

    Solved by:

        $ sudo apt-get install libglew-dev libcheese7 libcheese-gtk23 libclutter-gst-2.0-0 libcogl15 libclutter-gtk-1.0-0 libclutter-1.0-0

* **Install CuDNN 5.1**

    Step 1: Register an nvidia developer account and [download cudnn here](https://developer.nvidia.com/cudnn) (about 80 MB). You might need `nvcc --version` to get your cuda version.

    Step 2: Check where your cuda installation is. For most people, it will be `/urs/local/cuda/`.For me, it is `/usr/local/cuda-8.0/`. You can check it with `which nvcc`.

    Step 3: Copy the files:
```
    $ cd folder/extracted/contents
    $ sudo cp include/cudnn.h /usr/local/cuda-8.0/include
    $ sudo cp lib64/libcudnn* /usr/local/cuda-8.0/lib64
    $ sudo chmod a+r /usr/local/cuda/lib64/libcudnn*
```
    **Check version**

    You might have to adjust the path. See step 2 of the installation.

    `$ cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2`

    **Notes**

    When you get an error like

    `F tensorflow/stream_executor/cuda/cuda_dnn.cc:427] could not set cudnn filter descriptor: CUDNN_STATUS_BAD_PARAM`

    with TensorFlow, you might consider using CuDNN v4 instead of v5.

    **Ubuntu users who installed it via apt**: [http://askubuntu.com/a/767270/10425](http://askubuntu.com/a/767270/10425)

    **If it still cannot compile with `caffe`**. Try next:
```
    $ cd folder/extracted/contents
    $ sudo cp -P include/cudnn.h /usr/include
    $ sudo cp -P lib64/libcudnn* /usr/lib/x86_64-linux-gnu/
    $ sudo chmod a+r /usr/lib/x86_64-linux-gnu/libcudnn*
```

* **Install OpenCV 3.1**

        [compiler]$ sudo apt-get install build-essential
        [required]$ sudo apt-get install cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev
        [optional]$ sudo apt-get install python-dev python-numpy libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev

        $ cd ~/<my_working_directory>
        $ git clone https://github.com/opencv/opencv.git

        $ cd ~/<my_working_directory>/opencv
        $ mkdir <cmake_binary_dir> # any name you like
        $ cd <cmake_binary_dir>
        $ cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local ..
        $ make
        $ sudo make install

* **Adjust Makefile**

 		$ cd /path/to/caffe_root
		 $ cp Makefile.config.example Makefile.config

	Adjust Makefile.config:

	* uncomment to build with cuDNN, add the `include path` and `lib path` of your cuDNN.

		> **USE_CUDNN := 1**
		\# Whatever else you find you need goes here.
		INCLUDE_DIRS := \$(PYTHON_INCLUDE) /usr/local/include  **/usr/local/cuda/include**
		LIBRARY_DIRS := \$(PYTHON_LIB) /usr/local/lib /usr/lib  **/usr/local/cuda/lib64**

	* add `cuda installation directory`:
		> **CUDA_DIR := /usr/local/cuda-8.0**

	* uncomment to use OpenCV 3
		> **OPENCV_VERSION := 3**

	* Uncomment and set the right path -- to compile the matlab interface
		> \# MATLAB directory should contain the mex binary in /bin.
		**MATLAB_DIR := /usr/local/MATLAB/R2015b**

	* Uncomment and set the right path -- to use Python 3
		> **PYTHON_LIBRARIES := boost_python3 python3.4m
		PYTHON_INCLUDE := /usr/include/python3.4m \
		/usr/lib/python3/dist-packages/numpy/core/include**

* **Install caffe**.

        $ cd /path/to/caffe-root
        $ sudo make all
        $ sudo make test
        $ sudo make runtest

* **Install 'pycaffe' and 'matcaffe'**.

        $ sudo make pycaffe
        $ sudo make matcaffe
Check if pycaffe is installed successfully:
		$ python3
		>>> import sys
		>>> sys.path.append(/path/to/caffe-root/python/)
		>>> import caffe
* **Try Caffe**.

    * **MNIST**
        * Prepare dataset
                $ cd /path/to/cafee-root
                $ ./data/mnist/get_mnist.sh
                $ ./examples/mnist/create_mnist.sh
        * LeNet: the MNIST Classification Model
            The model has been defined in `./examples/mnist/lenet_train_test.prototxt`.
        * Define the MNIST solver
            The solver has been defined in `./examples/mnist/lenet_solver.prototxt`.
        * Training and testing the model
            One can do this after preparing dataset by
                $ ./examples/mnist/train_lenet.sh

    * **Cifar10**
        * Prepare dataset
                $ cd /path/to/cafee-root
                $ ./data/cifar10/get_cifar10.sh
                $ ./examples/cifar10/create_cifar10.sh
        * Model definition
            The model has been defined in `./examples/cifar10/cifar10_quick_train_test.prototxt`.
        * Define the MNIST solver
            The solver has been defined in `./examples/cifar10/cifar10_quick_solver.prototxt`.
        * Training and testing the model
            One can do this after preparing dataset by
                $ ./examples/cifar10/train_quick.sh

