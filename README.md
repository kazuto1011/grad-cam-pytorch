# Grad-CAM with PyTorch

PyTorch implementation of [Grad-CAM (Gradient-weighted Class Activation Mapping) [1]](https://arxiv.org/pdf/1610.02391v1.pdf). Grad-CAM localizes and highlights the discriminative regions that Convolutional Neural Networks-based models activate to predict visual concepts. This code contains only an imeplementation for the VGG-19 image classification models.

 - [x] Grad-CAM
 - [ ] Guided backpropagation and Guided Grad-CAM

## Dependencies
* Python 2.7
* PyTorch

## Usage
```bash
$ python generate.py [image]
```

## Examples
|Input|Cauliflower|Cucumber|Broccoli|Head cabbage|Zucchini|
|:-:|:-:|:-:|:-:|:-:|:-:|
|![](samples/vegetables.jpg)|![](samples/vegetables/cauliflower.png)|![](samples/vegetables/cucumber.png)|![](samples/vegetables/broccoli.png)|![](samples/vegetables/head_cabbage.png)|![](samples/vegetables/zucchini.png)|

## References
\[1\] R. R. Selvaraju, A. Das, R. Vedantam, M. Cogswell, D. Parikh, and D. Batra. "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization". arXiv, 2016
