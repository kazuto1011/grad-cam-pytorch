# Grad-CAM with PyTorch

PyTorch implementation of [Grad-CAM (Gradient-weighted Class Activation Mapping) [1]](https://arxiv.org/pdf/1610.02391v1.pdf). Grad-CAM localizes and highlights the discriminative regions that Convolutional Neural Networks-based models activate to predict visual concepts. This code contains only an imeplementation for the ResNet-152 image classification models.

## Dependencies
* Python 2.7
* PyTorch
* torchvision

## Usage
```bash
$ python main.py --image samples/cat_dog.jpg [--no-cuda]
```

## Examples
![](samples/cat_dog.png)

||bull mastiff|tiger cat|boxer|
|:-:|:-:|:-:|:-:|
|Probability|0.54285|0.19302|0.10428|
|**Grad-CAM [1]**|![](results/bull_mastiff_gcam.png)|![](results/tiger_cat_gcam.png)|![](results/boxer_gcam.png)|
|Vanilla Backpropogation|![](results/bull_mastiff_bp.png)|![](results/tiger_cat_bp.png)|![](results/boxer_bp.png)|
|Guided Backpropagation [2]|![](results/bull_mastiff_gbp.png)|![](results/tiger_cat_gbp.png)|![](results/boxer_gbp.png)|
|**Guided Grad-CAM [1]**|![](results/bull_mastiff_ggcam.png)|![](results/tiger_cat_ggcam.png)|![](results/boxer_ggcam.png)|

## References
\[1\] R. R. Selvaraju, A. Das, R. Vedantam, M. Cogswell, D. Parikh, and D. Batra. "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization". arXiv, 2016<br>
\[2\] J. T. Springenberg, A. Dosovitskiy, T. Brox, and M. Riedmiller. "Striving for Simplicity: The All Convolutional Net". arXiv, 2014