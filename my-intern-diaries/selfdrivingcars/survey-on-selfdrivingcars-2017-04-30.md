End-to-End learning for self-driving cars' steering
--

## 概述

目前，基于视觉的自动驾驶系统可以分成两类[3]：仲裁感知方法（*`mediated perception approaches`*）和行为映射方法（*`behavior reflex approaches`*）。仲裁感知方法对整个场景进行解析、理解，包括了多个子识别任务，比如道路识别、交通信号识别、车辆识别、行人识别等，然后将这些子任务的识别结果综合起来，通过一个智能仲裁中心，对车辆的行驶行为做出决策，这种方法有时也称为基于规则（*`rule-based`*）的方法。行为映射方法则直接将传感器的输入映射到驾驶行为（*`driving action`*），比如利用一个深度神经网络，将摄像头看到的场景映射为方向盘的转角（*`steering angle`*），这种方法也称为端到端（*`end-to-end`*）的方法。

Chen et al. 在他的文章中argue，仲裁感知方法的诸多子任务构成了一个非常复杂而冗余的输入空间，然而对于自动驾驶来说，只要正确地控制方向盘转角和车辆速度，车辆就可以正确而安全地在道路上行驶，因此输出空间是相当低维的。虽然仲裁感知方法目前是*`state-of-the-art`*，但仲裁感知方法还依赖于激光测距仪（*laser range finder*）、GPS、雷达和高精度地图，因而引入了许多难以解决的难题。事实上，对于自动驾驶决策来说，与其每一个决策相关的传感器和子识别任务是相当少的。

对于行为映射方法，Chen et al. argue, 有几个原因使得端到端的自动驾驶任务变得十分棘手。首先，当道路上有车时，即使在相同的场景，不同的驾驶员所做出的驾驶决策也可能完全不同，使得训练一个回归器变得十分困难。比如，当车辆在道路上笔直行驶时，有的司机可能会选择跟在前车后行驶，也可能会从左边超车，或者从右边超车。其次，行为映射方法的决策是很底层的。因为模型无法看到整个过程，比如当一辆车超车后又回到原来的轨道上，对于这辆车来说，仅仅是方向盘转角稍微发生了一系列变化，模型可能无法理解实际上发生了什么。最后，因为模型的输入是整张图片，学习算法必须学会决定图片的哪部分是与决策相关的，然而方向盘转角对训练的监督可能太弱而无法使学习算法能够正确地提取出图象中的关键信息。

对于采用卷积网络的端到端自动驾驶模型来说，端到端的方法确实存在着这些问题。从直观上看，由于卷积网络仅仅以当前信息作为输入， 并仅根据当前输入做出驾驶决策。当其看到一辆在笔直的道路上行驶的车突然转向时，往往无法理解发生了什么，因此仅仅采用卷积网络的端到端学习的固有缺陷是无法做出高层次的驾驶决策。但是如果一个模型能够记住之前的场景序列，比如一辆在笔直道路上行驶的车在数帧之前右转，当这辆车在前方无障碍物的情况下左转时，它就能理解，这辆车超车后准备回到原来的轨道上行驶，因此这个模型将可以做出超车后回到原来轨道上行驶的决策。因此我们认为，自动驾驶中的序列学习可以形成高层语义，从而理解并做出车辆驾驶过程的一系列决策，理解为什么相同的图片输入会产生截然不同的驾驶决策。

因此，本次调研主要关注于自动驾驶中的端到端学习模型，特别是序列学习模型。

#### 1. 相关工作介绍

从网络结构方面，根据目前的调研情况，可以大致将端到端的自动驾驶模型分成三类：基于卷积网络的模型、基于RNN的序列学习模型以及基于生成对抗网络（GAN)的生成学习模型。基于卷积网络的模型主要有 `ALVINN`、`DAVE`、`DeepDriving`和`DAVE2`等，基于RNN的序列学习模型有`FCN+LSTM`、`CNN+Attention-based LSTM`等，基于GAN的主要有`GAN+VAE+RNN`等。

`ALVINN`（*Autonomous Land Vehicle In A Network Network*）是利用神经网络进行自动驾驶的端到端学习的第一次尝试[1]。该模型用了一个3层的网络，包括输入层、隐藏层和输出层，以摄像头拍摄的图像和激光测距仪作为输入，并以行驶方向作为输出。最终训练出的模型可以以 `0.5 m/s` 的速度在CMU校园的一段400米的林间小道上正确行驶。

`DAVE` (*DARPA Autonomous Vehicle, DARPA, Defense Advanced Research Project Agency seedling project*) 是一个6层卷积网络[2]，以左右两个摄像头拍摄的图像作为网络输入，输出向左向右两个方向。该文章单目摄像头和双目摄像头进行了比较，发现采用双目摄像头对网络性能提升十分有限。该文献搭建了一辆小车，并通过无线电在室内对小车进行远程控制。

`DeepDriving`的想法稍有不同[3]。该文章提出了利用一个卷积网络先将输入图片映射到多个预测值，然后再利用一个驾驶控制器根据这些预测值计算出转向角和速度值。

`DAVE2` [4]基于`ALVINN`和`DAVE`，直接利用一个卷积网络，以左中右三个摄像头拍摄的图片作为输入，对方向盘转向角进行回归拟合。

文献[5]提出了一个`FCN-LSTM`的结构，该文章利用FCN网络对图像进行编码，得到的特征跟其对应的速度和转向角联合在一起，作为`LSTM`网络的输入，输出对下一个运动的预测。文献[6]提出利用`CNN`提取特征，然后输入到一个`attention-based`的`LSTM`中，对方向盘转角进行回归。

文献[7]则提出了一个`GAN+VAE+RNN`的生成学习模型。


#### 2. 本次调研主要工作

1. 准备数据集，从百度的roadhackers到comma.ai，编写数据输入层

1. 实现一个简单的 CNN 模型，验证数据输入层的正确性，初步验证神经网络用于自动驾驶方向盘转角预测的可行性

1. 实现一个 CNN(AlexNet) + LSTM，验证序列学习用于自动驾驶方向盘转角预测的可行性

1. 将CNN该成FCN，将LSTM改成Attention-based LSTM，并进行实验对比，是否能够进一步提高自动驾驶方向盘转角预测的准确度

1. 分析总结，未来方向

1. 研读相关文献


#### 3. 相关文献

[1] **alvinn1989** Pomerleau et al. *ALVINN: An Autonomous Land Vehicle In A Neural Network*. 1989, CMU.
[2] **dave2004** *DAVE: Autonomous Off-Road Vehicle Control Using End-to-End Learning*. 2004, DARPA.
[3] **deepdriving2015** Chen et al. *DeepDriving: Learning Affordance for Direct Perception in Autonomous Driving*. Dec 2015, Princeton University.
[4] **dave2016** Bojarski et al. *DAVE2: End-to-End Learning for Self-Driving Cars*. NVIDIA Corp.
[5] **gan-vae-rnn2016** Hotz et al. *Learning a Driving Simulator*. Aug 2016, comma.ai
[6] **fcn-lstm2016** Xu et al. *End-to-End Learning of Driving Models from Large-scale Video Datasets*. Dec 2016, UC Berkeley.
[7] **attention-lstm2017** Kim et al. *Interpretable Learning for Self-Driving Cars by Visualizing Causal Attention*. Mar 2017, UC Berkeley.


## 数据集介绍

#### 1. 百度 roadhackers

该数据集由百度的地图采集车采集得到。目前已经有数百万公里的采集数据，所[开源的数据](http://roadhackers.baidu.com/#downloads)中只有其中的一万公里（下载包1G $\approx 6.25 km$）。

采集的数据主要分成两大部分：一是百度全景图片数据，二是汽车姿态数据。

关于图片数据，地图采集车采集到的是360$^o$全景数据，由于数据量太大，开源的数据只是截取正前方$320\times 320$分辨率的图片，且图片下方与车头相切。

关于汽车姿态数据，主要公开了当前行驶速度，以及行驶轨迹的曲率。速度为当前时刻$t_0$的数据，分为$V_x$和$V_y$两个分量，分别表示向东以及向北（即坐标轴正方向）的速度。曲率为$t_0 + 0.125s$之后的数据。

#### 2. comma.ai

[Comma.ai 数据集](https://archive.org/details/comma-dataset)包含了10个不同大小的视频片段，由装在一辆Acura ILX 2016汽车挡风玻璃上的摄像头以20Hz的频率采集而来。 同时数据集还包含了车辆信息（车速、油门、方向盘、GPS坐标、陀螺仪角度信息等，100信息数据/s）。
（The dataset consists of 10 videos clips of variable size recorded at 20 Hz with a camera mounted on the windshield of an Acura ILX 2016. In parallel to the videos we also recorded some measurements such as car's speed, acceleration, steering angle, GPS coordinates, gyroscope angles. See the full log list here. These measurements are transformed into a uniform 100 Hz time base.）

数据集结构如下：

    +-- dataset
    |   +-- camera
    |   |   +-- 2016-04-21--14-48-08.h5
    |   |   ...
    |   +-- log
    |   |   +-- 2016-04-21--14-48-08.h5
    |   |   ...

所有的文件都是*hdf5*格式，并以其录制的时间命名。摄像头数据集的图像尺寸为 number_frames x 3 x 160 x 320，类型为*uint8*。log文件中的一项是 `cam1_ptr`，用于图像帧和测量信息之间的对齐。

一个log文件的数据结构如下。

其中以`app_`开头的关键字表示`Applanix POS LV 220E`，`X, Y, Z`坐标表示车辆前向、右向和下向三个方向。

以`fiber_`开头的关键字表示`KVH 1775 fiber-optic gyro`， `X, Y, Z`坐标表示车辆的右向45$^o$，左向45$^o$，以及向上。

以`velodyne_`开头的关键字表示 `HDL-32` 速度单位。

|HDF5 key |	Description|
|:-------:|:----------:|
|UN_D_cam1_ptr	|   |
|UN_D_cam2_ptr	|   |
|UN_D_lidar_ptr	|   |
|UN_D_radar_msg	|   |
|UN_D_rawgps	|   |
|UN_T_cam1_ptr	|   |
|UN_T_cam2_ptr	|   |
|UN_T_lidar_ptr	|   |
|UN_T_radar_msg	|   |
|UN_T_rawgps	|   |
|app_accel|	m/s^2, (Forward, Right, Down)|   |
|app_heading|	deg   |
|app_pos|	(Lat (deg), Lon (deg), Alt (m))   |
|app_speed	|m/s   |
|app_status|	   |
|app_v_yaw|	deg/s, yaw velocity   |
|blinker|	   |
|brake|	brake_computer + brake_user   |
|brake_computer|	Commanded brake [0-4095]   |
|brake_user|	User brake pedal depression [0-4095]   |
|cam1_ptr|   |
|Camera| frame index at this time   |
|cam2_ptr|	   |
|car_accel|	m/s^2, from derivative of wheel speed   |
|fiber_accel|	m/s^2   |
|fiber_compass|	   |
|fiber_compass_x|   |
|fiber_compass_y|	   |
|fiber_compass_z|	   |
|fiber_gyro|	deg/s   |
|fiber_temperature|	   |
|gas|	[0-1)   |
|gear_choice|	Selected gear. 0- park/neutral, 10- reverse, 11- gear currently changing   |
|idx	|   |
|rpm	|   |
|rpm_post_torque|	post torque-converter   |
|selfdrive	|   |
|speed|	m/s, from encoder after transmission, negative when gear is Revese   |
|speed_abs|	m/s, from encoder after transmission   |
|speed_fl|	Individual wheels speeds (m/s)   |
|speed_fr|	   |
|speed_rl	|   |
|speed_rr|	   |
|standstill|	Is the car stopped?   |
|steering_angle|	   |
|steering_torque|	deg/s, despite the name, this is the steering angle rate   |
|times|	seconds   |
|velodyne_gps|	   |
|velodyne_heading|	   |
|velodyne_imu|	   |


## 模型

#### 1. CNN

~~~mermaid
graph TD
	PythonLayer --> |image_data| Conv1[Conv1 --size 8x8 --num_out 16 --stride 4]
	PythonLayer --> |cont_indicator| Silence[Silence]
	PythonLayer --> |input_sequence| Silence
	PythonLayer --> |target_sequence| Loss[EuclideanLossLayer]

	Conv1 --> |conv1| Relu1[relu1]
	
	Relu1 --> |conv1| Conv2[Conv2 --size 5x5 --num_out 32 --stride 2]
	
	Conv2 --> |conv2| Relu2[relu2]
	
	Relu2 --> |conv2| Conv3[Conv3 --size 5x5 --num_out 64 --stride 2]
	
	Conv3 --> |conv3| FC1[InnerProduct --num_out 512]
	
	FC1 --> |fc4| FC2[InnerProduct --num_out 1]
	
	FC2 --> |predict| Loss
	
	Loss --> Lossout(mse)
~~~
#### 2. CaffeNet + LSTM
1. **模型 1**

	~~~mermaid
	graph TD
		PythonLayer --> |image_data| CaffeNet[CaffeNet --num_output 4096]

		CaffeNet --> |fc6| Reshapefc6[ReshapeLayer --dim 16 24 4096]
		Reshapefc6 --> |fc6_reshape| Concat[ConcatLayer --dim 16 24 4097]

		PythonLayer --> |cont_indicator| Reshapedcont[ReshapeLayer --dim 16 24]
		PythonLayer --> |input_sequence| ReshapedInput[ReshapeLayer --dim 16 24 1]
		PythonLayer --> |target_sequence| ReshapedTarget[RehapeLayer --dim 16 24]
		ReshapedTarget --> |target_sequence_reshape| Loss[EuclideanLossLayer]

		ReshapedInput --> |input_sequnce_reshape| Concat

		Concat --> |fc6_input_sequence| LSTM2[LSTMLayer --num_output 512]

		Reshapedcont --> |cont_indicator_reshape| LSTM2

		LSTM2 --> |lstm| InnerProduct[InnerProductLayer --num_output 1]

		InnerProduct --> |predict| Loss
		Loss --> Lossout(mse)
	~~~

1. **模型 2**
	~~~mermaid
	graph TD
		PythonLayer --> |image_data| CaffeNet[CaffeNet --num_output 4096]

		CaffeNet --> |fc6| Reshapefc6[ReshapeLayer --dim 16 24 4096]
		Reshapefc6 --> |fc6_reshape| Concat[ConcatLayer]

		PythonLayer --> |cont_indicator| Reshapedcont[ReshapeLayer --dim 16 24]
		Reshapedcont --> |cont_indicator_reshape| LSTM[LSTMLayer --num_output 128]
		PythonLayer --> |input_sequence| ReshapedInput[ReshapeLayer --dim 16 24 1]
		PythonLayer --> |target_sequence| ReshapedTarget[RehapeLayer --dim 16 24]
		ReshapedTarget --> |target_sequence_reshape| Loss[EuclideanLossLayer]

		ReshapedInput --> |input_sequence_reshape| LSTM
		LSTM --> |lstm1| Concat

		Concat --> |fc6_lstm1| LSTM2[LSTMLayer --num_output 512]

		Reshapedcont --> |cont_indicator_reshape| LSTM2

		LSTM2 --> |lstm2| InnerProduct[InnerProductLayer --num_output 1]

		InnerProduct --> |predict| Loss
		Loss --> Lossout(mse)

	~~~


#### 3. FCN + Attention-based LSTM


## 实验结果


## 总结与分析


## 未来工作

## 附录-相关模型介绍

## ALVINN - An Autonomous Land Vehicle In A Neural Network (1989@CMU)
ALVINN 

## DAVE - Autonomous Off-Road Vehicle Control Using End-to-End learning (2004)
    
Defense Advanced Research Project Agency (DARPA) seedling project 
DAVE - DARPA Autonomous Vehicle

整个项目历时7个月，由3~7个人组成的队伍完成，在硬件上的总共花费大概30000美元，包括了车辆、传感器、远程控制以及训练和存储数据用的电脑。

- 硬件结构

整个硬件结构包括PC机模块和小车模块。PC机根据游戏手柄的输入通过无线电远程操控小车采集数据，并训练自动驾驶模型。
    
- 自动驾驶模型

自动驾驶模型是一个六层卷积网络。输入层是降采样为$149\times 58$的原始图像， YUV色彩空间。输出层是表示向左、向右两个方向的两个$1\times 1$的特征图。
    
第一层 30 个 $3\times 3$的卷积核，降采样比例为$1\times 1$。
第二层 8 个 卷积核，降采样比例为$3\times 4$。
第三层 96 个 卷积核， 降采样比例为$1\times 1$。
第四层 24 个 卷积核， 降采样比例为$5\times 3$。
第五层 1920 个卷积核， 降采样比例为$1\times 1$。
第六层 600 个卷积核， 降采样比例为$1\times 1$。

## DeepDriving: Learning Affordance for Direct Perception in Autonomous Driving (2015)
    CNN based

## End-to-end Learning of Driving Models from Large-scale Video Datasets (2016)
    FCN-LSTM

## Learning a Driving Simulator (2016)
    GAN+VAE+RNN

## Query-efficient imitation learning for end-to-end autonomous driving (2016)
## End-to-end Learning for self-driving cars (2016)
    CNN
## Interpretable Learning for Self-Driving Cars by Visualizing Causal Attention (2017)
    Attention-based LSTM



