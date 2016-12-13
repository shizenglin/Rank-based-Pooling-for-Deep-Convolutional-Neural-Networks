# Rank-based-Pooling-for-Deep-Convolutional-Neural-Networks

Zenglin Shi, Yangdong Ye, Yunpeng Wu: Rank-based Pooling for Deep Convolutional Neural Networks.Neural Networks. 83: 21-31 (2016)

A novel pooling mechanism called rank-based pooling is proposed. We consider rank-based pooling as an instance of weighted pooling where a weighted sum of activations is used to generate the pooling output. Three new pooling methods, rank-based average pooling (RAP), rank-based weighted pooling (RWP) and rank-based stochastic pooling (RSP), are introduced according to different weighting strategies. RAP can be regarded as a tradeoff between max pooling and average pooling. A rank threshold t is used to eliminate some near-zero activations. The weights of those selected activations are set to be 1/t while others are kept as 0. In RWP, each activation is weighted by a coefficient in range of (0,1) computed based on its rank, and more important activations have larger weights. RSP replaces the conventional deterministic pooling operations with a stochastic procedure. A multinomial distribution is created from the probabilities p based on the ranks. The weights of the activations can be got by sampling from this distribution.

1.Rank-Based Average Pooling (RAP)

layer {

  name: "pool1"
  
  type: "Pooling"
  
  bottom: "conv1"
  
  top: "pool1"
  
  pooling_param {
  
    pool: RANKAVE
    
    kernel_size: 3
    
    stride: 2
    
    select_num: 8 
    
  }
  
}

select_num represents the t. RAP uses the parameter t to control how many activations are selected to be averaged. The value of t should not be too large or too small since t=1 corresponds to max-pooling, and t=n (pooling size) reduces to average-pooling. Optimal t may be different in different visual tasks, but we suggest that setting t at medium ranges (i.g., median value) may lead to satisfactory result.

2.Rank-based Weighted Pooling (RWP)

layer {

  name: "pool1"
  
  type: "Pooling"
  
  bottom: "conv1"
  
  top: "pool1"
  
  pooling_param {
  
    pool: RANKWEIGHTING
    
    kernel_size: 3
    
    stride: 2
    
    p_a: 0.5
    
  }
  
}

3.Rank-based Stochastic Pooling (RSP)

layer {

  name: "pool1"
  
  type: "Pooling"
  
  bottom: "conv1"
  
  top: "pool1"
  
  pooling_param {
  
    pool: RANKSTOCHASTIC
    
    kernel_size: 3
    
    stride: 2
    
    p_a: 0.5
    
  }
  
}

In RWP and RSP, we introduce a new hyper-parameter a(p_a). It controls the probability of the maximum activation in a pooling region. We argue that setting it to be around 0.5 may lead to satisfactory performance.

We evaluate the proposed methods on four benchmark datasets: MNIST, CIFAR-10, CIFAR-100 and NORB(Download: http://pan.baidu.com/s/1c2DQ4Ac psw:hgg8). All experiments are conducted by using a deep learning framework called Caffe. The proposed methods is simple and can be easily implemented.

In order to use the rank-based pooling in your caffe, some files need to be modified.

1.caffe.proto

In message PoolingParameter, adding 

enum PoolMethod {

    MAX = 0;
    
    AVE = 1;
    
    STOCHASTIC = 2;
    
    RANKSTOCHASTIC = 3;
    
    EXPSTOCHASTIC = 4;
    
    VALUEWEIGHTING = 5;
    
    RANKWEIGHTING = 6;
    
    RANKAVE = 7;
    
  }
  
 optional float p_a = 13 [default = 0.5];
  
 optional uint32 select_num = 14 [default = 5];
  
2.vision_layers.hpp

In PoolingLayer, adding

float p_a;

int select_num;

3.pooling_layer.cpp

In LayerSetUp, adding

if (this->layer_param_.pooling_param().pool() ==

      PoolingParameter_PoolMethod_RANKSTOCHASTIC) {
      
      p_a=pool_param.p_a();
      
  }
  
  if (this->layer_param_.pooling_param().pool() ==
  
      PoolingParameter_PoolMethod_RANKWEIGHTING) {
      
      p_a=pool_param.p_a();
      
  }
  
  if (this->layer_param_.pooling_param().pool() ==
  
      PoolingParameter_PoolMethod_RANKAVE) {
      
      select_num=pool_param.select_num();
      
  }
  
4.pooling_layer.cu

You can directly copy my pooling_layer.cu.

