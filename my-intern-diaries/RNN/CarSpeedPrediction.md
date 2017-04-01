~~~mermaid
graph TD

    PythonLayer --> |image_data| CaffeNet[CaffeNet --num_output 4096]
	
	CaffeNet --> |fc6| Reshapefc6[ReshapeLayer --dim 16 24 4096]
	Reshapefc6 --> |fc6_reshape| Concat[ConcatLayer]
	
	
	PythonLayer --> |cont_speed| LSTM[LSTMLayer --num_output 1000]
	PythonLayer --> |input_speed| Embed[Embed --input_dim 8000 --num_output 1000]
	PythonLayer --> |target_speed| Loss[SoftmaxWithLossLayer]
	
	Embed --> |embedded_input_speed| LSTM
	LSTM --> |lstm1| Concat
	
	Concat --> |fc6_lstm1| LSTM2[LSTMLayer --num_output 1000]
	
	PythonLayer --> |cont_speed| LSTM2
	
	LSTM2 --> |lstm2| InnerProduct[InnerProductLayer --num_output 8000]
	
	InnerProduct --> |predict| Loss
	Loss --> Lossout(cross_entropy_loss)
	
	InnerProduct --> |predict| Accuracy[AccuracyLayer]
	PythonLayer --> |target_speed| Accuracy
	Accuracy --> AccuracyOut(accuracy)
~~~