# Weighted Correlation Extension
This project provides a rudimentary PyTorch CUDA extension implementation of weighted correlation operation and layer, proposed by paper [Video Modeling with Correlation Networks](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_Video_Modeling_With_Correlation_Networks_CVPR_2020_paper.pdf). 

## Install

Make sure you have installed the following prerequisites:
* torch>=1.3, <=1.6
* gcc>=4.9 (and a maximum version restriction which varies depending on your CUDA version)
* nvcc

Install this project in development mode by

``pip3 install -e <path_to_project>``

## Restrictions

Up to now,
* Only CUDA mode is implemented. CPU mode is not supported yet.
* The kernel size K must be no greater than 15. 
* For `WeightedCorrelationLayerExtension` (i.e., perform correlation operation for each two consecutive sampled frames as in the paper), correlation operations are called iteratively in PyTorch API, not optimized by CUDA. So, efficiency might get worse with long input sequence. 

## Usage

The modules of `WeightedCorrelation` and `WeightedCorrelationLayerExtension` in `weighted_correlation.py` can be used in the same way as common Modules built by PyTorch API. 

## Performance

Compared with weighted correlation layer [implemented in pure PyTorch API](https://github.com/tefantasy/CorrNet), this CUDA implementation significantly reduces GPU memory consumption. 

Speed is approximately the same as PyTorch API implementation when clip length L=32. Higher relative speed is expected with smaller clip length. 

## Acknowledgements

**[Pytorch-Correlation-extension](https://github.com/ClementPinard/Pytorch-Correlation-extension)**
