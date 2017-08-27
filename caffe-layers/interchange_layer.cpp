#include <vector>

#include "caffe/layers/interchange_layer.hpp"

namespace caffe{

template <typename Dtype>
void InterchangeLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*> &bottom,
        const vector<Blob<Dtype>*> &top){
    CHECK_NE(bottom[0], top[0]) << this->type() << " Layer does not allow in-place computation.";
    CHECK_EQ(bottom[0]->num_axes(), 4) << "Number of axes of bottom blob must be 4.";
}

template <typename Dtype>
void InterchangeLayer<Dtype>::Reshape(const vector<Blob<Dtype>*> &bottom,
        const vector <Blob<Dtype>*> &top){
    
    const int num = bottom[0]->shape()[0];
    const int channels = bottom[0]->shape()[1];
    const int dim = bottom[0]->shape()[2] * bottom[0]->shape()[3];

    vector<int> top_shape(4);
    top_shape[0] = num;
    top_shape[1] = dim;
    int height = this->layer_param_.interchange_param().height();
    int width = this->layer_param_.interchange_param().width();

    CHECK_NE((width <= 0) && (height<= 0), true) << "height or width, at least one must be set " 
        << "for Layer " << this->type();

    if (height <= 0) height = channels / width;
    else if(width <= 0) width = channels / height;

    CHECK_EQ(width * height, channels) << "height * width != channels in Layer " 
        << this->type();
    top_shape[2] = height;
    top_shape[3] = width;
    top[0]->Reshape(top_shape);
}

template <typename Dtype>
void InterchangeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*> &bottom,
        const vector<Blob<Dtype>*> &top){
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();
    int num = bottom[0]->shape()[0];
    int channels = bottom[0]->shape(1);
    int dim = bottom[0]->shape(2) * bottom[0]->shape(3);
    
    for(int n = 0; n < num; ++n){
        for(int c = 0; c < channels; ++c){
            for(int d = 0; d < dim; ++d){
                top_data[d * channels + c] = bottom_data[c * dim + d];
            }
            top_data += dim * channels;
            bottom_data += dim * channels;
        }
    }
}

template <typename Dtype>
void InterchangeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*> &top,
        const vector<bool> &propagate_down, const vector<Blob<Dtype>*> &bottom){
        
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

    int num = bottom[0]->shape()[0];
    int channels = bottom[0]->shape(1);
    int dim = bottom[0]->shape(2) * bottom[0]->shape(3);

    if(this->param_propagate_down_[0]){
        for(int n = 0; n < num; ++n){
            for(int c = 0; c < channels; ++c){
                for(int d = 0; d < dim; ++d){
                    bottom_diff[c * dim + d] = top_diff[d * channels + c];
                }
                bottom_diff += dim * channels;
                top_diff += dim * channels;
            }
        }
    }

}

#ifdef CPU_ONLY
STUB_GPU(InterchangeLayer);
#endif

INSTANTIATE_CLASS(InterchangeLayer);
REGISTER_LAYER_CLASS(Interchange);

} //namespace caffe
