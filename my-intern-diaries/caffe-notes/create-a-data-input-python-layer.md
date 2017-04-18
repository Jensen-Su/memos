创建Python数据输入层
--

本节以**LRCN**(Long short-term recurrent convolutional net)的数据输入层为例，记录如何创建一个python数据输入层。

该数据输入层的任务是从$N$个视频中读取一个$T$帧的视频片段（*clip*)，同时读取每一帧对应的标记，然后将这些数据喂给LRCN网络进行预测或者训练。


## 数据集文件组织结构

```
    +--/path/to/caffe/data/
    |    +--dataset_name/
    |    |    video_list.txt: (video_name label(if any))
    |    |        122(video name) 2(label)
    |    |        139 3
    |    |        ...
    |    |    all.txt: (frame_path  label1 label2)
    |    |        ./frames/132.h5/132_0.jpg 13.56 -0.48
    |    |        ./frames/132.h5/132_1.jpg 13.61 -0.70
    |    |        ...
    |    |    +--train/
    |    |    |    train.txt: (video_name label)
    |    |    |        122 0.23
    |    |    |        122 0.24
    |    |    |        ...
    |    |    +--test/
    |    |    |    test.txt
    |    |    +--frames/
    |    |        122.txt
    |    |        132.txt
    |    |        ...
    |    |        +--122.h5/
    |    |        |    122_001.jpg
    |    |        |    122_002.jpg
    |    |        |    ...
    |    |        +--132.h5/
    |    |        |    132_001.jpg
    |    |        |    132_002.jpg
    |    |        |    ...
    |    |        ...
```

`dataset_name`是数据集的根目录名，在该数据集的根目录下有两个`.txt`文件和三个文件夹。`video_list.txt`列出了该数据集的所有视频以及各个视频的标签（如果有的话）。`all.txt`列出了所有帧（所有视频）的路径及其标签。文件夹`train/`和`test/`下分别有两个文件`train.txt`和`test.txt`，分别列出了用于训练和测试的视频。文件夹`frames/`下放置了各个视频的帧图片和标签文件，各个视频的所有帧图片放置在单独的文件夹下，标签文件放在`frames/`下。

## 数据输入层的任务

给定每次读取的批大小(*batch size*)$N$和每一个样本的帧数$T$，数据输入层的任务是：
1. 选定$N$个视频（随机选取？循环选取？）
2. 从这$N$个视频中分别截取$N$段长度为$T$帧的片段（随机截取？循环截取？）
3. 读取这些帧对应的标签
4. 输出图片数据（`image_data`)、序列输入标签(`input_sequence`)、序列目标标签(`targe_sequence`)、连续性指示器（`cont_indicator`)

其中输出数据的数据结构为:

    numpy.shape(image_data)     --> (T x N, 3, image_height, image_width)
    numpy.shape(input_sequence) --> (T x N,)
    numpy.shape(targe_sequence) --> (T x N,)
    numpy.shape(cont_indicator) --> (T x N,)

$T\timesN$意味着序列的组织形式应当为$[video_0-frame_0, video_1-frame_0, ...]$.

## A tutorial for "`python`" Layer in caffe


#### How to implement a "`python`" layer?
A "`python`" layer should be implemented as a python class derived from `caffe.Layer` base class. This class **must** have the following four methods:

```python
class my_py_layer(caffe.Layer):
    def setup(self, bottom, top):
        pass

    def reshape(self, bottom, top):
        pass

    def forward(self, bottom, top):
        pass

    def backward(self, top, propagate_down, bottom):
        pass

```

What are these methods?

`def setup(self, bottom, top):` THis method is called once when caffe builds the net. This function should check that number of inputs (`len(bottom)`) and number of outputs (`len(top)`) is as expected.

You should also allocate parameters of the net here (i.e., `self.add_blobs()`), see [this thread](http://stackoverflow.com/questions/39458461/writing-custom-python-layer-with-learnable-parameters-in-caffe) for more information.

This method has access to `self.param_str`, a string passed from the prototxt to the layer. See [this thread](http://stackoverflow.com/questions/34549743/caffe-how-to-get-the-phase-of-a-python-layer) for more information.

`def reshape(self, bottom, top):` This method is called whenever caffe reshapes the net. This function should allocate the outputs (each of the `top` blobs). The outputs's shape is usually related to the `bottom` s' shape.

`def forward(self, bottom, top):` Implementing the forward pass from `bottom` to `top`.

`def backward(self, top, propagate_down, bottom):` This method implements the backpropagation, it propagates the gradients from `top` to `bottom`. `propagate_down` is a Boolean vector of `len(bottom)` indicating to which of the `bottom`s the gradient should be propagated.

Some more information about `bottom` and `top` inputs you can find in [this post](http://stackoverflow.com/questions/33778225/building-custom-caffe-layer-in-python).


#### How to add a "`python`" layer in prototxt?

You need to add the following to your prototxt:
```prototxt
layer {
  name: 'rpn-data'
  type: 'Python'  
  bottom: 'rpn_cls_score'
  bottom: 'gt_boxes'
  bottom: 'im_info'
  bottom: 'data'
  top: 'rpn_labels'
  top: 'rpn_bbox_targets'
  top: 'rpn_bbox_inside_weights'
  top: 'rpn_bbox_outside_weights'
  python_param {
    module: 'rpn.anchor_target_layer'  # python module name where your implementation is
    layer: 'AnchorTargetLayer'   # the name of the class implementation
    param_str: "'feat_stride': 16"   # optional parameters to the layer
  }
```

#### How to set some parameters for a Python layer?

See [this thread](http://stackoverflow.com/questions/34549743/caffe-how-to-get-the-phase-of-a-python-layer)
If we would like to insert a "`python`" layer but only include it in some phases, for example, only include it in "`train`" phase:
```prototxt
layer {
  name: "my_py_layer"
  type: "Python"
  bottom: "in"
  top: "out"
  python_param {
    module: "my_module_name"
    layer: "myLayer"
  }
  include { phase: TRAIN } # THIS IS THE TRICKY PART!
}
```

how can our python layer know this?

This can be done using the param_str parameters of the python_param.
Here's how it's done:
```prototxt
layer {
  type: "Python"
  ...
  python_param {
    ...
    param_str: '{"phase":"TRAIN","numeric_arg":5}' # passing params as a STRING
```
In python, we get param_str in the layer's setup function:

```python
import caffe, json
class myLayer(caffe.Layer):
  def setup(self, bottom, top):
    param = json.loads( self.param_str ) # use JSON to convert string to dict
    self.phase = param['phase']
    self.other_param = int( param['numeric_arg'] ) # I might want to use this as wel
```

## 写一个复杂的具有数据预取功能的python输入层


1. `def setup(self. bottom, top):` 

    ```python

    class VideoInputLayer(caffe.Layer):
        def setup(self, bottom, top):
            ##-- 1. check the number of inputs and the number of outputs
            if len(bottom) > 0:
                raise Exception('Cannot have bottoms for data input layer!')
            self.top_names = ['image_data', 'input_sequence', 'targe_sequence', 'cont_indicator']
            if len(top) != len(self.top_names):
                raise Exception('Incorrect number of outputs (expected %d, got %d)'%
                        (len(self.top_names), len(top)))
            self.reshape(bottom, top)

            ##-- 2. open file `all.txt`, construct a dictionary with frame paths as keys and lables as values
            all_frames_file = open('all.txt', 'r')
            all_frames_lines = all_frames_file.readlines()
            all_frames_file.close()
            self.frame_dict = {}
            for line in lines:
                frame_path = line.split(' ')[0]
                self.frame_dict[frame_path] = []
                for label in line.split(' ')[1:]:
                    self.frames_dict[frame_path].append(float(label))

            ##-- 3. construct a dictionary for videos
            all_videos_file = open('video_list.txt', 'r')
            all_videos_lines = all_videos_file.readlines()
            all_videos_file.close()

            self.video_dict = {}
            self.video_order = []
            # for each video
            for idx, line in enumerate(all_videos_lines):
                video_name = line.split(' ')[0][:-1]
                video_label = None
                if len(line.split(' ')) > 1:
                    video_label = int(line.split(' ')[1])

                video_frames_file = open(video_name + '.txt')
                video_frames_lines = video_frames_file.readlines()
                video_frames_file.close()
                
                video_frames = []
                for line in video_frames_lines:
                    video_frames.append(line.split(' ')[0])

                self.video_dict[video_name] = {}
                self.video_dict[video_name]['name'] = video_name
                self.video_dict[video_name]['label'] = video_label
                self.video_dict[video_name]['length'] = len(video_frames)
                self.video_dict[video_name]['frames'] = video_frames
                self.video_dict[video_name]['reshape'] = (240, 320)
                self.video_dict[video_name]['crop'] = (227, 227)
                self.video_dict[video_name]['ptr'] = np.random.randint(self.num_all_frames - self.clip_length - 1)
                self.video_order.append(video_name)

            ##-- 4. creat and setup transformer for the input data
            shape = (self.num_all_frames, self.num_image_channels, self.image_height, self.image_width)
            self.transformer = caffe.io.Transformer({'data_in' : shape})
            image_mean = [0.0, 0.0, 0.0]
            channel_mean = np.zeros((3, 227, 227))
            for channel_index, mean_val in enumerate(image_mean):
                channel_mean[channel_index, ...] = mean_val

            # rescale from [0, 1] to [0, 255]
            self.transformer.set_raw_scale('data_in', 255)
            # substract the dataset-mean value in each channel
            self.transformer.set_mean('data_in', channel_mean)
            # swap channels from RGB to BGR
            self.transformer.set_channel_swap('data_in', (2, 1, 0))
            # move the image channels to outermost dimension
            self.transformer.set_transpose('data_in', (2, 0, 1))

            ##-- 5. creat thread pool for data fetching
            self.thread_result = {}
            self.thread = None
            pool_size = 24 # thread pool size, i.e., the number of threads

            # create a image processor
            self.image_processor = ImageProcessor(self.transformer)
            # create a sequence generator
            self.sequence_generator = SequenceGenerator(self.N, self.video_dict, self.frames_dict)
            # create a thread pool
            self.pool = Pool(processes = pool_size)
            # create a batch fetcher
            self.batch_fetcher = BatchFetcher(self.thread_result, self.sequence_generator,
                    self.image_processor, self.pool)

            ##-- 6. begin data fetching
            self.dispatch_worker()
            ## self.dispatch_worker() --> self.batch_advancer() -->
            ## advance_batch() --> 
            ##           sequence_generator() --> sequenceGeneratorVideo() 
            ##           image_processor() --> ImageProcessorCrop() --> processImageCrop()
            self.join_worker()


    ```
1. `def reshape(self, bottom, top):`

    ```python
    def reshape(self, bottom, top):
        print "==================== Reshaping data ==============="
        for top_index, name in enumerate(self.top_names):
            if name == 'image_data':
                shape = (self.num_all_frames, self.num_image_channels, self.image_height, self.image_width)
            else:
                shape = (self.num_all_frames, )

            top[top_index].reshape(*shape)

            print "reshape data blob top['" + name + "'] as" + str(shape)
    
    ```

1. `def forward(self, bottom, top):`
    ```python
    def forward(self, bottom, top):
        if self.thread is not None:
            self.join_worker()
        
        #rearrange the data: The LSTM takes inputs as [video0_frame0, video1_frame0,...] but the data is currently arranged as [video0_frame0, video0_frame1, ...]
        new_image_data = [None]*len(self.thread_result['image_data'])
        
        new_input_sequence = [None]*len(self.thread_result['input_sequence']) 
        new_target_sequence = [None]*len(self.thread_result['target_sequence'])
        new_cont_indicator = [None]*len(self.thread_result['cont_indicator'])
        for i in range(self.clip_length):
          for ii in range(self.batch_size):
            old_idx = ii*self.clip_length + i
            new_idx = i*self.batch_size + ii
            new_image_data[new_idx] = self.thread_result['image_data'][old_idx]
            new_input_sequence[new_idx] = self.thread_result['input_sequence'][old_idx]
            new_target_sequence[new_idx] = self.thread_result['target_sequence'][old_idx]
            new_cont_indicator[new_idx] = self.thread_result['cont_indicator'][old_idx]

        for top_index, name in zip(range(len(top)), self.top_names):
          if name == 'image_data':
            for i in range(self.num_all_frames):
              top[top_index].data[i, ...] = new_image_data[i] 
          elif name == 'input_sequence':
            top[top_index].data[...] = new_input_sequence
          elif name == 'target_sequence':
            top[top_index].data[...] = new_target_sequence 
          elif name == 'cont_indicator':
            top[top_index].data[...] = new_cont_indicator
        
        # prefetching the next batch
        self.dispatch_worker()
    ```

1. `def backward` pass.

1. `dispatch_worker` and `join_worker`

    ```python
    def dispatch_worker(self):
        assert self.thread is None
        self.thread = Thread(target=self.batch_fetcher)
        # would call `BatchFetcher.__call__(self)`
        self.thread.start()

    def join_worker(self):
        assert self.thread is not None
        self.thread.join()
        self.thread = None

    ```

1. `BatchFetcher.__call__(self):`
    ```python
    class BatchAdvancer():
        def __init__(self, result, sequence_generator, image_processor, pool):
            self.result = result
            self.sequence_generator = sequence_generator
            self.image_processor = image_processor
            self.pool = pool

        def __call__(self):
            """
            Load and process each of the frames. crop, reshape, flip.
            Add cont_indicator.
            ----
            `result`: result['image_data'], result['input_sequence'], 
                      result['target_sequence'], result['cont_indicator']
            """
            # call `SequenceGenerator.__call__(self)` to get a batch of frames
            input_sequence, target_sequence, frames_info = self.sequence_generator()

            # frames_info[0]: path
            # call image_processor(frames_info)
            self.result['image_data'] = self.pool.map(self.image_processor, frames_info)
            
            self.result['input_sequence'] = input_sequence
            self.result['target_sequence'] = target_sequence
            cont = np.ones(len(input_sequence))
            cont[0::clip_length] = 0
            self.result['cont_indicator'] = cont
    ```

1. `SequenceGenerator.__call__(self):`

    ```python
    class SequenceGenerator(object):
        def __init__(self, batch_size, video_dict, frames_annotation):
            self.batch_size = batch_size
            self.clip_length = clip_length
            self.num_all_frames = self.batch_size * self.clip_length
            self.video_dict = video_dict
            self.frames_annotation = frames_annotation
            self.video_order = self.video_dict.keys()
            self.num_videos = len(self.video_dict.keys())
            self.idx = 0

        def __call__(self):
            """
            return a tuple `(label_r, im_info)` where
            * `label_r` is a list of labels for each frame, its shape is:
            `[clip0_frame0_label, clip0_frame1_label, ..., clip1_frame0_label, ...]`
            * `im_info` is a tuple `(im_paths, im_crop, im_reshape, im_flip)` for each frame.
            `im_path`: [clip0_frame0_path, clip0_frame1_path, ...]
            `im_crop`: [clip0_frame0_crop, clip0_frame1_crop, ...]
            `im_reshape`: [clip0_frame0_reshape, clip0_frame1_reshape, ...]
            `im_flip`: [clip0_frame0_flip, clip0_frame1_flip, ...]
            """
            input_sequence = []
            target_sequence = []
            frame_paths = []
            frame_crops = []
            frame_reshapes = []
            frame_flips = []

            ## 循环选取视频
            if self.idx + self.batch_size >= self.num_videos:
                idx_list = range(self.idx, self.num_videos)
                idx_list.extend(range(0, self.batch_size - (self.num_videos - self.idx)))
            else:
                idx_list = range(self.idx, self.idx + self.batch_size)
            
            for i in idx_list:
                # for each video
                video_name = self.video_order[i]
                label = self.video_dict[video_name]['label']
                crop = self.video_dict[video_name]['crop']
                reshape = self.video_dict[video_name]['reshape']

                # 随机开始，循环截取视频片段
                # decide the begin frame of the clip to be extracted
                clip_begin = self.video_dict[video_name]['ptr']
                video_length = self.video_dict[video_name]['length']
                if clip_begin + self.clip_length > video_length -1:
                    clip_begin = np.random.randint(video_length - self.clip_length - 1)
                    self.video_dict[video_name]['ptr'] = clip_begin
                
                # extract a clip of length `self.clip_length` from `clip_begin`
                video_frames = self.video_dict[video_name]['frames']
                for k in range(clip_begin, clip_begin + self.clip_length):
                    frame_paths.append(video_frames[k])
                    value = self.frames_annotation[video_frames[k]]
                    input_sequence.append(value)
                    if k != clip_begin:
                        target_sequence.append(value)
                
                value = self.frames_annotation[video_frames[clip_begin + self.clip_length]]
                target_sequence.append(value)
                
                frame_reshapes.extend([(reshape)]* self.clip_length)
                r0 = int(np.random.random() * (reshape[0] - crop[0]))
                r1 = int(np.random.random() * (reshape[1] - crop[1]))

                frame_crops.extend([(r0, r1, r0 + crop[0], r1 + crop[1])] * self.clip_length)
                frame_flips.extend([np.random.randint(0, 2)] * self.clip_length)
            
                # 下一次选取本视频段时，从上次截取的视频片段后面开始截取
                self.video_dict[video_name]['ptr'] = clip_begin + clip_length

            frames_info = zip(frame_paths, frame_crops, frame_reshapes, frame_flips)
            # 循环选取视频
            self.idx += self.batch_size
            if self.idx >= self.num_videos:
                self.idx = self.idx - self.num_videos
            
            return input_sequence, target_sequence, frames_info

    ```
