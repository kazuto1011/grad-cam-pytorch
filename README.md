# Grad-CAM with PyTorch

PyTorch implementation of [Grad-CAM (Gradient-weighted Class Activation Mapping) [1]](https://arxiv.org/pdf/1610.02391v1.pdf). Grad-CAM localizes and highlights the discriminative regions that Convolutional Neural Networks-based models activate to predict visual concepts. This code contains only an imeplementation for the VGG-19 image classification models.

## Dependencies
* Python 2.7
* PyTorch

## Usage
```bash
$ python generate.py --image samples/vegetables.jpg
```

## Examples
![](samples/vegetables.jpg)

||Cauliflower|Cucumber|Broccoli|Head cabbage|Zucchini|
|:-:|:-:|:-:|:-:|:-:|:-:|
|**Grad-CAM [1]**|![](results/gcam_cauliflower.png)|![](results/gcam_cucumber.png)|![](results/gcam_broccoli.png)|![](results/gcam_head_cabbage.png)|![](results/gcam_zucchini.png)|
|Backpropogation|![](results/bp_cauliflower.png)|![](results/bp_cucumber.png)|![](results/bp_broccoli.png)|![](results/bp_head_cabbage.png)|![](results/bp_zucchini.png)|
|Guided Backpropagation [2]|![](results/gbp_cauliflower.png)|![](results/gbp_cucumber.png)|![](results/gbp_broccoli.png)|![](results/gbp_head_cabbage.png)|![](results/gbp_zucchini.png)|
|**Guided Grad-CAM [1]**|![](results/ggcam_cauliflower.png)|![](results/ggcam_cucumber.png)|![](results/ggcam_broccoli.png)|![](results/ggcam_head_cabbage.png)|![](results/ggcam_zucchini.png)|

## References
\[1\] R. R. Selvaraju, A. Das, R. Vedantam, M. Cogswell, D. Parikh, and D. Batra. "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization". arXiv, 2016<br>
\[2\] J. T. Springenberg, A. Dosovitskiy, T. Brox, and M. Riedmiller. "Striving for Simplicity: The All Convolutional Net". arXiv, 2014