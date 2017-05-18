# Grad-CAM with PyTorch

PyTorch implementation of [Grad-CAM (Gradient-weighted Class Activation Mapping) [1]](https://arxiv.org/pdf/1610.02391v1.pdf)

## Dependencies
* Python 2.7
* PyTorch

## Usage
```bash
$ python grad_cam.py [image]
```

## Examples
|Input|Cauliflower|Cucumber|Broccoli|Head cabbage|Zucchini|
|:-:|:-:|:-:|:-:|:-:|:-:|
|![](samples/vegetables.jpg)|![](results/grad_cam_cauliflower.png)|![](results/grad_cam_cucumber.png)|![](results/grad_cam_broccoli.png)|![](results/grad_cam_head_cabbage.png)|![](results/grad_cam_zucchini.png)|

## References
[1] R. R. Selvaraju _et al._ "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization". arXiv, 2016