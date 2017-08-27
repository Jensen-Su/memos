#ifndef CAFFE_INTERCHANGE_LAYER_HPP_
#define CAFFE_INTERCHANGE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"

namespace caffe{
    
/**
 * @brief Interchange the role of feature map and channels,
 * i.e., each position in a feature map is changed into a channel.
 */
template <typename Dtype>
class InterchangeLayer: public Layer<Dtype> {
public:
    explicit InterchangeLayer(const LayerParameter &param)
        : Layer<Dtype>(param) {}
    virtual void LayerSetUp(const vector<Blob<Dtype>*> &bottom,
            const vector<Blob<Dtype>*> &top);
    virtual void Reshape(const vector<Blob<Dtype>*> &bottom,
            const vector<Blob<Dtype>*> &top);

    virtual inline const char *type() const { return "Interchange"; }
    virtual inline int ExactNumBottomBlobs() const { return 1; }
    virtual inline int ExactNumTopBlobs() const { return 1; }

protected:

    virtual void Forward_cpu(const vector<Blob<Dtype>*> &bottom,
            const vector<Blob<Dtype>*> &top);
    virtual void Forward_gpu(const vector<Blob<Dtype>*> &bottom,
            const vector<Blob<Dtype>*> &top);

    virtual void Backward_cpu(const vector<Blob<Dtype>*> &top,
            const vector<bool> &propagate_down, const vector<Blob<Dtype>*> &bottom);
    virtual void Backward_gpu(const vector<Blob<Dtype>*> &top,
            const vector<bool> &propagate_down, const vector<Blob<Dtype>*> &bottom);
}; // class Interchangelayer
} // namespace caffe


#endif // CAFFE_INTERCHANGE_LAYER_HPP_
