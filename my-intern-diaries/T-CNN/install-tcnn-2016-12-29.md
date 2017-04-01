# Installing T-CNN.

### Prerequisites
1. [caffe](http://caffe.berkeleyvision.org) with `Python layer` and `pycaffe`
2. [GNU Parallel](http://www.gnu.org/software/parallel/)
3. [matutils](https://github.com/myfavouritekk/matutils)
4. [FCN tracker](https://github.com/scott89/FCNT)
5. `Matlab` with python [engine](http://www.mathworks.com/help/matlab/matlab-engine-for-python.html?refresh=true)

### Instructions

0. Before installing T-CNN

    ```bash
        # install GNU Parallel
        $ sudo apt-get install parallel
        $ cd "matlabroot/extern/engines/python"
        # install matlab.engine
        $ python setup.py install
    ```

1. Clone the repository and sub-repositories from GitHub, let `$TCNN_ROOT` represents the root directory of the repository.

    ```bash
        $ # clone the repository
        $ git clone --recursive https://github.com/myfavouritekk/T-CNN.git
        $ cd $TCNN_ROOT
        $ # checkout the ImageNet 2015 VID branch
        $ git checkout ilsvrc2015vid
    ```

2. Compilation for `vdetlib`

    ```bash
        $ cd $TCNN_ROOT/vdetlib
        $ make
        $ export PYTHONPATH=$TCNN_ROOT/vdetlib:$PYTHONPATH
    ```
3. Download and install `caffe` in the `External` directory

    ```bash
        $ git clone https://github.com/BVLC/caffe.git External/caffe
        $ # modify `Makefile.config` and build with Python layer and pycaffe
        $ # detailed instruction, please follow http://caffe.berkeleyvision.org/installation.html
        $ export PYTHONPATH=$TCNN_ROOT/External/caffe/python:$PYTHONPATH
    ```

4. Download a modified version of [`FCN Tracker`](https://github.com/myfavouritekk/FCNT/tree/T-CNN) originally developed by Lijun Wang et. al.

* Clone the repository, and let `$FCNT_ROOT` represents the root directory.
    
    ```bash
        $ git clone --recursive -b T-CNN https://github.com/myfavouritekk/FCNT fcn_tracker_matlab
    ```

* Compile the corresponding `caffe-fcn_tracking`.

    ```bash
        $ cd $FCNT_ROOT/caffe-fcn_tracking
        $ # modify Makefile.config and compile with matcaffe
        $ make all && make matcaffe
        $ # copy compiled +caffe folder in to root folder
        $ cp -r matlab/+caffe $FCNT_ROOT
    ```
    **NOTE**: if you are installing a *caffe* version which assumes you are using an older *cudnn* version but you have installed a newer *cudnn*, you may get errors like:

    ```bash
    In file included from ./include/caffe/util/device_alternate.hpp:40:0,  
                     from ./include/caffe/common.hpp:19,  
                     from src/caffe/common.cpp:7:  
    ./include/caffe/util/cudnn.hpp: In function ‘void caffe::cudnn::createPoolingDesc(cudnnPoolingStruct**, caffe::PoolingParameter_PoolMethod, cudnnPoolingMode_t*, int, int, int, int, int, int)’:  
    ./include/caffe/util/cudnn.hpp:127:41: error: too few arguments to function ‘cudnnStatus_t cudnnSetPooling2dDescriptor(cudnnPoolingDescriptor_t, cudnnPoolingMode_t, cudnnNanPropagation_t, int, int, int, int, int, int)’  
             pad_h, pad_w, stride_h, stride_w));  
                                             ^  
    ./include/caffe/util/cudnn.hpp:15:28: note: in definition of macro ‘CUDNN_CHECK’  
         cudnnStatus_t status = condition; \  
                                ^  
    In file included from ./include/caffe/util/cudnn.hpp:5:0,  
                     from ./include/caffe/util/device_alternate.hpp:40,  
                     from ./include/caffe/common.hpp:19,  
                     from src/caffe/common.cpp:7:  
    /usr/local/cuda-7.5//include/cudnn.h:803:27: note: declared here  
     cudnnStatus_t CUDNNWINAPI cudnnSetPooling2dDescriptor(  
                               ^  
    make: *** [.build_release/src/caffe/common.o] Error 1
    ```

    One quick way to work around this is to:

    ```bash
        # replace the cudnn.hpp with the newest version
        $ cp $NEWEST_CAFFE_ROOT/include/caffe/util/cudnn.hpp ./include/caffe/util/cudnn.hpp
        # replace the cudnn* with the newest version
        $ cp $NEWEST_CAFFE_ROOT/src/caffe/layers/cudnn* ./src/caffe/layers/
        $ cp $NEWEST_CAFFE_ROOT/include/caffe/layers/cudnn_* ./include/caffe/layers/
    ```
    
* Compile the `mexResize` code in `DSST_code`. Copy the compiled mex function to `$FCNT_ROOT`.

    ```matlab
        >> cd 'DSST_code'
        >> compilemex
    ```

* Compile and copy [`gradientMex`](https://github.com/pdollar/toolbox/blob/master/channels/private/gradientMex.cpp) fucntion from Piotr Dollar's [toolbox](https://github.com/pdollar/toolbox) into `$FCNT_ROOT`. Some pre-compiled functions for Linux, OSX and Windows are included.

* Download `ImageNet` pre-trained `VGG` models from [here](https://gist.github.com/ksimonyan/211839e770f7b538e2d8) and put it in `$FCNT_ROOT`.

    ```bash
        $ cd $FCNT_ROOT
        $ wget http://www.robots.ox.ac.uk/%7Evgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel -O VGG_ILSVRC_16_layers.caffemodel
    ```

* Run the demo code in `Matlab`.

    ```matlab
        >> demo
    ```
    **NOTE**: I got a `out-of-memory` dump in my workstation which owns a TITAN X GPU with 12GB memory.

## Demo
1. Extract the sample data and still-image detection results

    ```bash
        $ cd $TCNN_ROOT
        $ unzip sample_data.zip -d data/
    ```

2. Generate optical flow for the videos

    ```bash
        $ mkdir ./data/opt_flow
        $ ls ./data/frames |
            parallel python tools/data_proc/gen_optical_flow.py ./data/frames/{} ./data/opt_flow/{} --merge
    ```
    
    **NOTE**: I got a problem with this line:
    ```bash
        $ ls ./data/frames |
            parallel python tools/data_proc/gen_optical_flow.py ./data/frames/{} ./data/opt_flow/{} --merge
        usage: gen_optical_flow.py [-h] [--bound BOUND] [--merge] [--debug]
                                   vid_dir save_dir
        gen_optical_flow.py: error: unrecognized arguments: ILSVRC2015_val_00007011
        $ ls ./data/frames
        ILSVRC2015_val_00007011/
    ```

    Since the result of `ls ./data/frames` is `ILSVRC2015_val_00007011`, which will be passed to `{}` via pipe. How about trying this:

    ```bash
        $ ls ./data/frames |
            parallel python tools/data_proc/gen_optical_flow.py ./data/frames/ILSVRC2015_val_00007011 ./data/opt_flow/ILSVRC2015_val_00007011 --merge
        parallel: Warning: Input is read from the terminal. Only experts do this on purpose. Press CTRL-D to exit.
        
    ```

    Anyhow I just could not get rid of this error if keep using `parallel`. So I quit using `parallel`. Another problem showed up:

    ```bash
        $ python tools/data_proc/gen_optical_flow.py ./data/frames/ILSVRC2015_val_00007011 ./data/opt_flow/ILSVRC2015_val_00007011 --merge
        Processing ./data/frames/ILSVRC2015_val_00007011: 140 files... 
        Traceback (most recent call last):
          File "tools/data_proc/gen_optical_flow.py", line 51, in <module>
            0.5, 3, 15, 3, 7, 1.5, 0)
        TypeError: integer argument expected, got float
    ```
    To workaround this bug, open file `gen_optical_flow.py`, do some small change:

    ```bash
        $ gvim tools/data_proc/gen_optical_flow.py
    ```
    Change the following code block
    ```python
        ...
        # normalize image size
        flow = cv2.calcOpticalFlowFarneback(
            cv2.resize(img1, None, fx=fxy, fy=fxy),
            cv2.resize(img2, None, fx=fxy, fy=fxy), 
            0.5, 3, 15, 3, 7, 1.5, 0)
    ```
    to (add a parameter `None` as following)

    ```python
        ...
        # normalize image size
        flow = cv2.calcOpticalFlowFarneback(
            cv2.resize(img1, None, fx=fxy, fy=fxy),
            cv2.resize(img2, None, fx=fxy, fy=fxy), None, 
            0.5, 3, 15, 3, 7, 1.5, 0)
    ```

3. Multi-context suppression and motion-guided propagation in **Matlab**

    ```matlab
        >> addpath(genpath('tools/mcs_mgp'));
        >> mcs_mgp('data/opt_flow', 'data/scores', 'data/mcs_mgp')
    ```

4. Tubelet tracking and re-scoring

    ```bash
        $ # generate .vid protocol files
        $ ls data/frames | parallel python vdetlib/tools/gen_vid_proto_file.py {} $PWD/data/frames/{} data/vids/{}.vid
        $ # tracking from raw detection files
        $ find data/vids -type f -name *.vid | parallel -j1 python tools/tracking/greedy_tracking_from_raw_dets.py {} data/mcs_mgp/window_size_7_time_step_1_top_ratio_0.000300_top_bonus_0.400000_optflow/{/.} data/tracks/{/.} --thres 3.15 --max_frames 100 --num 30
        $ # spatial max-pooling
        $ find data/vids -type f | parallel python tools/scoring/tubelet_raw_dets_max_pooling.py {} data/tracks/{/.} data/mcs_mgp/window_size_7_time_step_1_top_ratio_0.000300_top_bonus_0.400000_optflow/{/.} data/score_proto/window_size_7_time_step_1_top_ratio_0.000300_top_bonus_0.400000_optflow_max_pooling/{/.} --overlap_thres 0.5
    ```

5. Tubelet visualization

    ```bash
        $ python tools/visual/show_score_proto.py data/vids/ILSVRC2015_val_00007011.vid data/score_proto/window_size_7_time_step_1_top_ratio_0.000300_top_bonus_0.400000_optflow_max_pooling/ILSVRC2015_val_00007011/ILSVRC2015_val_00007011.airplane.score
    ```

## Beyond demo
1. Optical flow extraction

    ```bash
        $ python tools data_proc/gen_optical_flow.py -h
    ```

2. [vdetlib](https://github.com/myfavouritekk/vdetlib) for tracking and rescoring
3. Visualization tools in `tools/visual`.


## Known Issues
1. Matlab engines may stall after long periods of tracking. Please consider to kill the certain matlab session to continue.

## To-do list
- [ ] Tubelet Bayesian classifier


