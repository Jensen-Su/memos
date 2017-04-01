#!/usr/bin/env python

"""
Data input layer for self-driving cars model.
---
Files of dataset should organised as following:

    -/path/to/caffe/data/
        -baidu_roadhackers/
            video_list.txt:
                122(video name) 2(label)
                139 3
                ...
            speed_curvature.txt:
                ./frames/132.h5/132_0.jpg 13.56 -0.48
                ./frames/132.h5/132_1.jpg 13.61 -0.70
                ...
            -train
                train.txt:
                    122 0.23
                    122 0.24
                    ...
            -test
                test.txt
            -frames
                122.txt
                132.txt
                ...
                -122.h5
                    122_001.jpg
                    122_002.jpg
                    ...
                -139.h5
                    139_001.jpg
                    139_002.jpg
                    ...
                ...
"""

import sys
sys.path.append('./python')
import caffe
import numpy as np
import glob
from multiprocessing import Pool
from threading import Thread

data_root_dir = '../../data/baidu_roadhackers/'
frames_dir = 'frames/'

train_dir = 'train/'
test_dir = 'test/'
frames_annotation_filepath = 'speed_curvature.txt'
train_list_filename = 'train.txt'
test_list_filename = 'test.txt'

clip_length = 16
train_batch_size = 4
test_batch_size  = 3

value_resolution = 1000

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
        
        sequence = []
        frame_paths = []
        frame_crops = []
        frame_reshapes = []
        frame_flips = []
        max_min = self.frames_annotation['max-min']

        if self.idx + self.batch_size >= self.num_videos:
            idx_list = range(self.idx, self.num_videos)
            idx_list.extend(range(0, self.batch_size - (self.num_videos - self.idx)))
        else:
            idx_list = range(self.idx, self.idx + self.batch_size)

        for i in idx_list:
            video_name = self.video_order[i]
            label = self.video_dict[video_name]['label']
            crop = self.video_dict[video_name]['crop']
            reshape = self.video_dict[video_name]['reshape']

            clip_begin = self.video_dict[video_name]['idx']
            video_length = self.video_dict[video_name]['length']
            if clip_begin + self.clip_length > video_length:
                clip_begin = np.random.randint(video_length - self.clip_length - 1)
                self.video_dict[video_name]['idx'] = clip_begin

            video_frames = self.video_dict[video_name]['frames']
            for k in range(clip_begin, clip_begin + self.clip_length):
                frame_paths.append(video_frames[k])
                value = self.frames_annotation[video_frames[k]]
                value = int(value * value_resolution / max_min)
                sequence.append(value)
            
            input_sequence = sequence
            target_sequence = sequence[1:]
            value = self.frames_annotation[video_frames[clip_begin + self.clip_length]]
            value = int(value * value_resolution / max_min)
            target_sequence.append(value)
            
            frame_reshapes.extend([(reshape)]* self.clip_length)
            r0 = int(np.random.random() * (reshape[0] - crop[0]))
            r1 = int(np.random.random() * (reshape[1] - crop[1]))
            frame_crops.extend([(r0, r1, r0 + crop[0], r1 + crop[1])] * self.clip_length)
            frame_flips.extend([np.random.randint(0, 2)] * self.clip_length)
        
        ###################################################################################
        ## open for test
        # print '\nfirst few items of sequence and frames_info'
        # print '-------------------------------------------'
        # for i in range(len(input_sequence)):
        #     print 'speed: %d' % input_sequence[i]
        #     print 'frame%d_path: %s' % (i, frame_paths[i])
        #     print 'frame%d_crop: (%d, %d, %d, %d)' % (i, frame_crops[i][0], frame_crops[i][1],
        #             frame_crops[i][2], frame_crops[i][3])
        #     print 'frame%d_reshape: (%d, %d)' % (i, frame_reshapes[i][0], frame_reshapes[i][1])
        #     print 'frame%d_flip: %d\n' % (i, frame_flips[i])

        frames_info = zip(frame_paths, frame_crops, frame_reshapes, frame_flips)
        
        self.idx += self.batch_size
        if self.idx >= self.num_videos:
            self.idx = self.idx - self.num_videos
        
        return input_sequence, target_sequence, frames_info



class ImageProcessor(object):
    def __init__(self, transformer):
        self.transformer = transformer

    def __call__(self, frames_info):
        frame_paths = frames_info[0]
        frame_crops = frames_info[1]
        frame_reshapes = frames_info[2]
        frame_flips = frames_info[3]

        data_in = caffe.io.load_image(frame_paths)
        if (data_in.shape[0] < frame_reshapes[0]) | (data_in.shape[1] < frame_reshapes[1]):
            data_in = caffe.io.resize(data_in, frame_reshapes)

        if frame_flips:
            # data_in = caffe.io.flip_image(data_in, flow)
            data_in = data_in[frame_crops[0]: frame_crops[2], frame_crops[1]: frame_crops[3], :]
        
        processed_image = self.transformer.preprocess('data_in', data_in)

        return processed_image

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
        input_sequence, target_sequence, frames_info = self.sequence_generator()

        # frames_info[0]: path
        # call image_processor(frames_info)
        self.result['image_data'] = self.pool.map(self.image_processor, frames_info)
        
        self.result['input_sequence'] = input_sequence
        self.result['target_sequence'] = target_sequence
        cont = np.ones(len(input_sequence))
        cont[0::clip_length] = 0
        self.result['cont_indicator'] = cont
        
        ############################################
        ## open for debug
        # print "self.result['image_data'][0]:"
        # print self.result['image_data'][0]
        # print 'input_sequence:'
        # print input_sequence
        # print 'cont_indicator:'
        # print cont
        ###########################################


class VideoInputLayer(caffe.Layer):

    def initialize(self):
        self.train_or_test = 'test'
        self.batch_size = test_batch_size
        self.clip_length = clip_length
        self.num_all_frames = self.batch_size * self.clip_length
        self.num_image_channels = 3
        self.height = 227
        self.width = 227
        self.video_list_filename = test_list_filename
        self.frames_annotation_filepath = frames_annotation_filepath

    def setup(self, bottom, top):
    # def setup(self):
        """
        This method is called once when caffe builds the net. 
        --
        1. This function should check that number of inputs (`len(bottom)`) 
           and number of outputs (`len(top)`) is as expected.
        2. You should also allocate internal parameters of the net here (i.e. `self.add_blobs()`)
        3. This method has access to `self.param_str`
        """
        if len(bottom) > 0:
            raise Exception('cannot have bottoms for data input layer!')

        # np.random.seed(10)
        self.initialize()

        ## open annotation file
        frames_annotation_file_path = data_root_dir + self.frames_annotation_filepath
        frames_annotation_file = open(frames_annotation_file_path, 'r')
        lines = frames_annotation_file.readlines()
        frames_annotation_file.close()
        self.frames_annotation = {}
        sequence_min = 100000
        sequence_max = 0
        for line in lines:
            # ./frames/126.h5/126_0.jpg 12.32 0.234
            value = np.abs(float(line.split(' ')[1]))
            self.frames_annotation[data_root_dir + line.split(' ')[0][2:]] = value
            if sequence_min > value:
                sequence_min = value
            if sequence_max < value:
                sequence_max = value
        self.frames_annotation['max-min'] = sequence_max - sequence_min

        ## test
        print 'Read annotation file'
        print '---------------------'
        key = self.frames_annotation.keys()[0]
        print 'First line: %s, %f\n' % (key, self.frames_annotation[key])

        ## open video list, read it and save all lines in `lines`
        video_list_path = data_root_dir + self.train_or_test + '/' +  self.video_list_filename
        video_list_file = open(video_list_path, 'r')
        lines = video_list_file.readlines()
        video_list_file.close()

        video_dict = {}
        current_line = 0
        video_order = []

        ## Construct a dictionary for videos
        for idx, line in enumerate(lines):
        # each line in lines is `video_name  video_label`
            split = line.split(' ')
            video_name = split[0]
            if len(split) > 1:
                video_label = int(split[1])
            else:
                video_label = -1
                video_name = video_name[:-2]

            # the directory where frames of the video `video_name` are put
            video_frames_dir = data_root_dir + frames_dir + video_name
            video_frames_dir = video_frames_dir[:-1]
            
            # video_frams, a list of paths of all frames: video_frames_dir/*.jpg
            # video_frames = glob.glob(r'%s/*.jpg' % video_frames_dir)
            frame_list_file = open(data_root_dir + frames_dir + video_name.split('.')[0] + '.txt')
            file_lines = frame_list_file.readlines()
            frame_list_file.close()
            video_frames = []
            for line in file_lines:
                frame_path = line.split(' ')[0]
                video_frames.append(data_root_dir + frame_path[2:])
            
            number_of_frames = len(video_frames)

            video_dict[video_name] = {}
            video_dict[video_name]['name'] = video_name
            video_dict[video_name]['label'] = video_label
            video_dict[video_name]['length'] = number_of_frames
            video_dict[video_name]['frames'] = video_frames
            video_dict[video_name]['reshape'] = (240, 320)
            video_dict[video_name]['crop'] = (227, 227)
            # idx: begin index of a video clip
            video_dict[video_name]['idx'] = np.random.randint(self.num_all_frames - self.clip_length - 1)

            video_order.append(video_name)

        self.video_dict = video_dict
        self.video_order = video_order
        self.num_videos = len(video_dict.keys())

        ## test if the video dictionary is constructed correctly
        video_items = video_dict.items()
        video = video_items[0]
        print '\nFirst item of video dictionary'
        print '------------------------------'
        print 'video name: %s' % video_items[0][0]
        item = video_items[0][1]
        print 'label: %d' % item['label']
        print 'length: %d frames' % item['length']
        print 'path of the first few frames:\n %s' % item['frames'][0]
        for i in range(1, 100):
            print item['frames'][i]
        ## creat and set up transformer for the input called 'data_in'
        shape = (self.num_all_frames, self.num_image_channels, self.height, self.width)
        self.transformer = caffe.io.Transformer({'data_in' : shape})
        image_mean = [103.939, 116.779, 128.68]
        channel_mean = np.zeros((3, 227, 227))

        for channel_index, mean_val in enumerate(image_mean):
            channel_mean[channel_index, ...] = mean_val
        
        # rescale from [0, 1] to [0, 255]
        self.transformer.set_raw_scale('data_in', 255) 
        # substract the dataset-mean value in each channel
        self.transformer.set_mean('data_in', channel_mean) 
        # swap channels from RGB to BGR
        self.transformer.set_channel_swap('data_in', (2, 1, 0)) 
        # move image channels to outermost dimension
        self.transformer.set_transpose('data_in', (2, 0, 1))  


        self.thread_result = {}
        self.thread = None
        pool_size = 24 # thread pool size, i.e., the number of threads

        self.image_processor = ImageProcessor(self.transformer)
        self.sequence_generator = SequenceGenerator(self.batch_size, self.video_dict, self.frames_annotation)

        self.pool = Pool(processes = pool_size)
        self.batch_advancer = BatchAdvancer(self.thread_result, self.sequence_generator,
                self.image_processor, self.pool)

        ## self.dispatch_worker() --> self.batch_advancer() -->
        ## advance_batch() --> 
        ##           sequence_generator() --> sequenceGeneratorVideo() 
        ##           image_processor() --> ImageProcessorCrop() --> processImageCrop()
        ## return self.thread_result containing data, label, and clip_markers
        self.dispatch_worker()

        self.top_names = ['image_data', 'input_sequence', 'target_sequence', 'cont_indicator']
        print 'Outputs:', self.top_names
        if len(top) != len(self.top_names):
          raise Exception('Incorrect number of outputs (expected %d, got %d)' %
                          (len(self.top_names), len(top)))
        self.reshape(bottom, top)

        self.join_worker()

    def reshape(self, bottom, top):
        print "============== Reshape data ================================"
        for top_index, name in enumerate(self.top_names):
          if name == 'image_data':
            shape = (self.num_all_frames, self.num_image_channels, self.height, self.width)
          elif name == 'input_sequence':
            shape = (self.num_all_frames,)
          elif name == 'target_sequence':
            shape = (self.num_all_frames,)
          elif name == 'cont_indicator':
            shape = (self.num_all_frames,)
          top[top_index].reshape(*shape)

    def forward(self, bottom, top):
    # def forward(self):
        if self.thread is not None:
            self.join_worker()

        #rearrange the data: The LSTM takes inputs as [video0_frame0, video1_frame0,...] but the data is currently arranged as [video0_frame0, video0_frame1, ...]
        new_image_data = [None]*len(self.thread_result['image_data'])
        
        ###########################
        ## open for test
        # print '\nnew_image_data:'
        # print '---------------'
        # print self.thread_result['image_data'][0] 

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
              top[top_index].data[i, ...] = new_result_data[i] 
          elif name == 'input_sequence':
            top[top_index].data[...] = new_input_sequence
          elif name == 'target_sequence':
            top[top_index].data[...] = new_target_sequence 
          elif name == 'cont_indicator':
            top[top_index].data[...] = new_cont_indicator

        self.dispatch_worker()

    def dispatch_worker(self):
        ## self.dispatch_worker() --> self.batch_advancer() -->
        ## advance_batch() --> 
        ##           sequence_generator() --> sequenceGeneratorVideo() 
        ##           image_processor() --> ImageProcessorCrop() --> processImageCrop()
        ## return self.thread_result containing data, label, and clip_markers
        assert self.thread is None
        self.thread = Thread(target=self.batch_advancer)
        self.thread.start()

    def join_worker(self):
        assert self.thread is not None
        self.thread.join()
        self.thread = None

    def backward(self, top, propagate_down, bottom):
        pass


class VideoInputLayerTest(VideoInputLayer):

    def initialize(self):
        self.train_or_test = 'test'
        self.batch_size = test_batch_size
        self.clip_length = clip_length
        self.num_all_frames = self.batch_size * self.clip_length
        self.num_image_channels = 3
        self.height = 227
        self.width = 227
        self.video_list_filename = test_list_filename
        self.frames_annotation_filepath = frames_annotation_filepath

class VideoInputLayerTrain(VideoInputLayer):

    def initialize(self):
        self.train_or_test = 'train'
        self.batch_size = train_batch_size 
        self.clip_length = clip_length
        self.num_all_frames = self.batch_size * self.clip_length
        self.num_image_channels = 3
        self.height = 227
        self.width = 227
        self.video_list_filename = train_list_filename 
        self.frames_annotation_filepath = frames_annotation_filepath
