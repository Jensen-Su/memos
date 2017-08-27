#include <vector>

#include "caffe/layers/interchange_layer.hpp"

namespace caffe{

// transpose a matrix using the diagonal method.
// see http://www.cs.colostate.edu/~cs675/MatrixTranspose.pdf
#define TILE_DIM 32 
#define BLOCK_ROWS 8
template <typename Dtype>
__global__ void transposeDiagonal(Dtype *odata,
        const Dtype *idata, int width, int height)
{
     /* int xIndex = blockIdx.x*TILE_DIM + threadIdx.x; */
     /* int yIndex = blockIdx.y*TILE_DIM + threadIdx.y; */
     /* int index_in = xIndex + width * yIndex; */
     /* int index_out = yIndex + height * xIndex; */
     /* for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS) { */
     /*     odata[index_out+i] = idata[index_in+i*width]; */
     /* } */
    __shared__ Dtype tile[TILE_DIM][TILE_DIM+1];
    int blockIdx_x, blockIdx_y;
    // diagonal reordering
    if (width == height) {
        blockIdx_y = blockIdx.x;
        blockIdx_x = (blockIdx.x+blockIdx.y)%gridDim.x;
    } else {
        int bid = blockIdx.x + gridDim.x*blockIdx.y;
        blockIdx_y = bid%gridDim.y;
        blockIdx_x = ((bid/gridDim.y)+blockIdx_y)%gridDim.x;
    }
    int xIndex = blockIdx_x*TILE_DIM + threadIdx.x;
    int yIndex = blockIdx_y*TILE_DIM + threadIdx.y;
    int index_in = xIndex + (yIndex)*width;
    xIndex = blockIdx_y*TILE_DIM + threadIdx.x;
    yIndex = blockIdx_x*TILE_DIM + threadIdx.y;
    int index_out = xIndex + (yIndex)*height;
    for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS) {
        tile[threadIdx.y+i][threadIdx.x] =
            idata[index_in+i*width];
    }

    __syncthreads();

    for (int i=0; i<TILE_DIM; i+=BLOCK_ROWS) {
        odata[index_out+i*height] =
            tile[threadIdx.x][threadIdx.y+i];
    }
} 
template <typename Dtype>
void InterchangeLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*> &bottom,
        const vector<Blob<Dtype>*> &top){

    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype*  top_data = top[0]->mutable_gpu_data();

    const int num = bottom[0]->shape(0);
    const int channels = bottom[0]->shape(1);
    const int dim = bottom[0]->shape(2) * bottom[0]->shape(3);
    dim3 grid(dim/TILE_DIM, channels/TILE_DIM);
    dim3 threads(TILE_DIM,BLOCK_ROWS);
    for(int n = 0; n < num; ++n){
        transposeDiagonal<Dtype> <<<grid, threads>>> (top_data, bottom_data, dim, channels);
        bottom_data += dim * channels;
        top_data += dim * channels;
    }
}

template <typename Dtype>
    void InterchangeLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*> &top,
            const vector<bool> &propagate_down, const vector<Blob<Dtype>*> &bottom){

        const Dtype* top_diff = top[0]->gpu_diff();
        Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();

        const int num = top[0]->shape(0);
        const int channels = top[0]->shape(1);
        const int dim = top[0]->shape(2) * top[0]->shape(3);
        dim3 grid(dim/TILE_DIM, channels/TILE_DIM);
        dim3 threads(TILE_DIM,BLOCK_ROWS);

        if(this->param_propagate_down_[0]){
            for(int n = 0; n < num; ++n){
                transposeDiagonal<Dtype> <<<grid, threads>>> (bottom_diff, top_diff, dim, channels);
                bottom_diff += dim * channels;
                top_diff += dim * channels;
            }
        }
    }

    INSTANTIATE_LAYER_GPU_FUNCS(InterchangeLayer);
} // namespace caffe
