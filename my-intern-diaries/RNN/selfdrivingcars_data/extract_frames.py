#!/usr/bin/env python
import h5py
import cv2
import os
import glob
import numpy as np

"""
Directories under `dataset_root_dir` should be organized as following:
    -dataset_root_dir
        extract_frames.py
        -camera
            -199.h5
            -122.h5
            ...
        -log
            -199.h5
            -122.h5
            ...
"""

# dataset_root_dir = "/home/jcsu/workshop/rnn/caffe-jeff/data/baidu_roadhackers/"
dataset_root_dir = "./"
sampling_interval = 1
sampling_num = None

def extract_frames(video_name, dataset_root_dir, sampling_interval, sampling_num):
    """
    video_name, for example: '119.h5'
    """
    camera_filename = './camera/' + video_name
    log_filename = './log/' + video_name
    video_frames_dir = dataset_root_dir + 'frames/' + video_name + '/'

    if not os.path.exists(video_frames_dir):
        os.mkdir(video_frames_dir)


    ## ./frames/119.txt
    sequence_split_txtfile = open(dataset_root_dir + 'frames/' + video_name.split('.')[0] + '.txt', 'w')
    sequence_txtfile = open(dataset_root_dir + 'speed_curvature.txt', 'w')


    logfile = h5py.File(log_filename, 'r')
    camerafile = h5py.File(camera_filename, 'r')

    logs = logfile['attrs'][:]
    ## ommit the first and last 100 frames
    start_idx = 100
    end_idx = len(logs) - 200

    if not sampling_num:
        sampling_num = (end_idx - start_idx) / sampling_interval

    idx_list = np.array(range(sampling_num))
    idx_list = idx_list * sampling_interval
    
    x_speed_list = logs[idx_list, 1]
    y_speed_list = logs[idx_list, 2]
    speed_list = np.sqrt(x_speed_list * x_speed_list + y_speed_list * y_speed_list)
    curvature_list = logs[idx_list, 3] * 1000 

    for i in range(len(idx_list)):
        image_idx = str('%0.3f' % logs[idx_list[i], 0])
        image_data = camerafile[image_idx]
        x1 = np.array(image_data[:, :, 2])
        x2 = np.array(image_data[:, :, 1])
        x3 = np.array(image_data[:, :, 0])
        image = cv2.merge([x1, x2, x3])

        image_path = video_frames_dir + video_name.split('.')[0] + '_' + str(idx_list[i]) + '.jpg'

        # save the frame
        cv2.imwrite(image_path, image)

        # save the speed & curvature
        # sequence_split_txtfile.write(image_path + ' ' + str(x_speed_list[i]) + ' ' + 
        #         str(y_speed_list[i]) + ' ' + str(curvature_list[i]) + '\n')

        # sequence_txtfile.write(image_path + ' ' + str(x_speed_list[i]) + ' ' + 
        #         str(y_speed_list[i]) + ' ' + str(curvature_list[i]) + '\n')
        # sequence_split_txtfile.write(image_path + ' ' + str(int(speed_list[i])) + ' ' + 
        #         str(int(curvature_list[i])) + '\n')

        # sequence_txtfile.write(image_path + ' ' + str(int(speed_list[i])) + ' ' + 
        #         str(int(curvature_list[i])) + '\n')
        sequence_split_txtfile.write(image_path + ' ' + str((speed_list[i])) + ' ' + 
                str((curvature_list[i])) + '\n')

        sequence_txtfile.write(image_path + ' ' + str((speed_list[i])) + ' ' + 
                str((curvature_list[i])) + '\n')

    sequence_split_txtfile.close()
    sequence_txtfile.close()
    logfile.close()
    camerafile.close()
    return sampling_num



if __name__ == "__main__":

    video_h5filenames = glob.glob('./camera/*.h5')
    
    video = video_h5filenames[0]
    for video in video_h5filenames:

        print 'video h5df file: ' + video
        print '-----------------------'
        print 'extracting frames, one frame from every %d frames...' % sampling_interval
        video_name = video.split('/')[-1]
        sampling_num = extract_frames(video_name, dataset_root_dir, sampling_interval, sampling_num )

        print 'extracted %d frames from %s \n' % (sampling_num, video)

        

