# Grad-CAM with PyTorch

PyTorch implementation of [Grad-CAM (Gradient-weighted Class Activation Mapping) [1]](https://arxiv.org/pdf/1610.02391v1.pdf). Grad-CAM localizes and highlights discriminative regions that convolutional neural networks-based models activate to predict visual concepts. This repo contains only an implementation for image classification models.

## Dependencies
* Python 2.7
* PyTorch 0.2.0
* torchvision
* click
* opencv

## Usage

```sh
python main.py --help
```

* ```--image-path```: a path to an image (required)
* ```--arch```: a model name from ```torchvision.models```, e.g., 'resnet152' (required)
* ```--topk```: the number of classes to generate (default: 3)
* ```--cuda/--no-cuda```: GPU or CPU

The command above generates, for top *k* classes:
* Vanilla backproped gradients
* Guided backproped gradients
* Deconvolved gradients
* Grad-CAM
* Guided Grad-CAM

The guided-* do not support F.relu but only nn.ReLU in this codes.
For instance, off-the-shelf *inception_v3* cannot cut off negative gradients during backward operation (#2).

## Examples

![](samples/cat_dog.png)

```layer4.2``` of ```torchvision.models.resnet152```

||bull mastiff|tiger cat|boxer|
|:-:|:-:|:-:|:-:|
|Probability|0.54285|0.19302|0.10428|
|Grad-CAM [1]|![](results/bull_mastiff_gcam_resnet152.png)|![](results/tiger_cat_gcam_resnet152.png)|![](results/boxer_gcam_resnet152.png)|
|Vanilla backpropagation|![](results/bull_mastiff_bp_resnet152.png)|![](results/tiger_cat_bp_resnet152.png)|![](results/boxer_bp_resnet152.png)|
|"Deconvnet" [2]|![](results/bull_mastiff_deconv_resnet152.png)|![](results/tiger_cat_deconv_resnet152.png)|![](results/boxer_deconv_resnet152.png)|
|Guided backpropagation [2]|![](results/bull_mastiff_gbp_resnet152.png)|![](results/tiger_cat_gbp_resnet152.png)|![](results/boxer_gbp_resnet152.png)|
|Guided Grad-CAM [1]|![](results/bull_mastiff_ggcam_resnet152.png)|![](results/tiger_cat_ggcam_resnet152.png)|![](results/boxer_ggcam_resnet152.png)|

Grad-CAM visualization of *bull mastiff*

|Model|resnet152|vgg19|vgg19_bn|inception_v3|densenet201|
|:-:|:-:|:-:|:-:|:-:|:-:|
|Input|224x224|224x224|224x224|299x299|224x224|
|Grad-CAM [1]|![](results/bull_mastiff_gcam_resnet152.png)|![](results/bull_mastiff_gcam_vgg19.png)|![](results/bull_mastiff_gcam_vgg19_bn.png)|![](results/bull_mastiff_gcam_inception_v3.png)|![](results/bull_mastiff_gcam_densenet201.png)|

## References

\[1\] R. R. Selvaraju, A. Das, R. Vedantam, M. Cogswell, D. Parikh, and D. Batra. "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization". arXiv, 2016<br>
\[2\] J. T. Springenberg, A. Dosovitskiy, T. Brox, and M. Riedmiller. "Striving for Simplicity: The All Convolutional Net". arXiv, 2014