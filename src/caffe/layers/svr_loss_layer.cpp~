#include <algorithm>
#include <cfloat>
#include <cmath>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SVRLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  const Dtype* label = bottom[1]->cpu_data();
  int num = bottom[0]->num();
  int count = bottom[0]->count();
  Dtype epsilon = this->layer_param_.svr_loss_param().epsilon();

  caffe_copy(count, bottom_data, bottom_diff);
  
  for (int i = 0; i < num; ++i) {
      bottom_diff[i] = std::max(
        Dtype(0), std::abs(label[i] - bottom_diff[i]) - epsilon);
  }
  Dtype* loss = top[0]->mutable_cpu_data();
  switch (this->layer_param_.svr_loss_param().norm1()) {
  case SVRLossParameter_Norm_L1:
    loss[0] = caffe_cpu_asum(count, bottom_diff) / num;
    break;
  case SVRLossParameter_Norm_L2:
    loss[0] = caffe_cpu_dot(count, bottom_diff, bottom_diff) / num;
    break;
  default:
    LOG(FATAL) << "Unknown Norm";
  }
}

template <typename Dtype>
void SVRLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    int num = bottom[0]->num();
    int count = bottom[0]->count();

    for (int i = 0; i < num; ++i) {
      bottom_diff[i] *= ((bottom[1]->cpu_data()[i] > bottom[0]->cpu_data()[i])?-1:1);
    }

    const Dtype loss_weight = top[0]->cpu_diff()[0];
    switch (this->layer_param_.svr_loss_param().norm1()) {
    case SVRLossParameter_Norm_L1:
      caffe_cpu_sign(count, bottom_diff, bottom_diff);
      caffe_scal(count, loss_weight / num, bottom_diff);
      break;
    case SVRLossParameter_Norm_L2:
      caffe_scal(count, loss_weight * 2 / num, bottom_diff);
      break;
    default:
      LOG(FATAL) << "Unknown Norm";
    }
  }
}

INSTANTIATE_CLASS(SVRLossLayer);
REGISTER_LAYER_CLASS(SVRLoss);

}  // namespace caffe
