#### Activity Recognition
``train_test_lstm_RGB.prototxt``
~~~mermaid
graph TD
	A[PythonLayer]--> |data| B[AlexNet --num_output 4096]
	A--> | labels | C[ReshapeLayer --dim 16 24/3]
	A--> | clip_marker | D[ReshapeLayer --dim 16 24/3]
	B --> |fc6| E[ReshapeLayer --dim 16 24/3 4096]
	E --> |fc6-reshape| F[LSTMLayer --num_output 256]
	D --> |reshape-cm| F
	F --> |lstm1| G[DropoutLayer]
	G --> |lstm1-drop| H[InnerProductLayer --num_output 101 --axis 2]
	H --> |fc8-final| I[SoftmaxLossLayer --axis 2]
	C --> |reshape-label| I
	H --> |fc8-final| J[AccuracyLayer --axis 2]
	C --> |reshape-label| J
	I --> K(loss)
	J --> L(accuracy)
~~~


#### Image Captioning

``lrcn.prototxt``lrcn_caffenet_to_lstm

~~~protobuf
layer {
  name: "data"
  type: "HDF5Data"
  top: "cont_sentence"
  top: "input_sentence"
  top: "target_sentence"
  include { phase: TRAIN }
  hdf5_data_param {
    source: "./examples/coco_caption/h5_data/buffer_100/train_unaligned_batches/hdf5_chunk_list.txt"
    batch_size: 20
  }
}
layer {
  name: "data"
  type: "HDF5Data"
  top: "cont_sentence"
  top: "input_sentence"
  top: "target_sentence"
  include {
    phase: TEST
    stage: "test-on-train"
  }
  hdf5_data_param {
    source: "./examples/coco_caption/h5_data/buffer_100/train_unaligned_batches/hdf5_chunk_list.txt"
    batch_size: 20
  }
}
layer {
  name: "data"
  type: "HDF5Data"
  top: "cont_sentence"
  top: "input_sentence"
  top: "target_sentence"
  include {
    phase: TEST
    stage: "test-on-val"
  }
  hdf5_data_param {
    source: "./examples/coco_caption/h5_data/buffer_100/val_unaligned_batches/hdf5_chunk_list.txt"
    batch_size: 20
  }
}
~~~


~~~mermaid
graph TD

	ImageDataLayer --> |data| CaffeNet
	CaffeNet[CaffeNet --num_output 1000] --> |fc8| LSTM2[LSTMLayer --num_output 1000]
	ImageDataLayer --> |label| Silence
	
	HDF5DataLayer --> |cont_sentence| LSTM[LSTMLayer --num_output 1000]
	HDF5DataLayer --> |input_sentence| Embed[Embed --input_dim 8801 --num_output 1000]
	HDF5DataLayer --> |target_sentence| Loss[SoftmaxWithLossLayer]
	
	Embed --> |embedded_input_sentence| LSTM
	LSTM --> |lstm1| LSTM2
	HDF5DataLayer --> |cont_sentence| LSTM2
	
	LSTM2 --> |lstm2| InnerProduct[InnerProductLayer --num_output 8801]
	
	InnerProduct --> |predict| Loss
	Loss --> Lossout(cross_entropy_loss)
	
	InnerProduct --> |predict| Accuracy[AccuracyLayer]
	HDF5DataLayer --> |target_sentence| Accuracy
	Accuracy --> AccuracyOut(accuracy)
~~~
