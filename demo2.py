#!/usr/bin/env python
# coding: utf-8
#
# Author: Kazuto Nakashima
# URL:    https://kazuto1011.github.io
# Date:   05 February 2019


from __future__ import print_function

import copy

import click
import cv2
import numpy as np
import torch
from torch.autograd import Variable
from torchvision import models, transforms

from grad_cam import GradCAM


def save_gradcam(filename, gcam, raw_image):
    h, w, _ = raw_image.shape
    gcam = cv2.resize(gcam, (w, h))
    gcam = cv2.applyColorMap(np.uint8(gcam * 255.0), cv2.COLORMAP_JET)
    gcam = gcam.astype(np.float) + raw_image.astype(np.float)
    gcam = gcam / gcam.max() * 255.0
    cv2.imwrite(filename, np.uint8(gcam))


@click.command()
@click.option("-i", "--image-path", type=str, required=True)
@click.option("--cuda/--no-cuda", default=True)
def main(image_path, cuda):
    device = torch.device("cuda" if cuda and torch.cuda.is_available() else "cpu")

    if cuda:
        current_device = torch.cuda.current_device()
        print("Running on the GPU:", torch.cuda.get_device_name(current_device))
    else:
        print("Running on the CPU")

    # Synset words
    classes = list()
    with open("samples/synset_words.txt") as lines:
        for line in lines:
            line = line.strip().split(" ", 1)[1]
            line = line.split(", ", 1)[0].replace(" ", "_")
            classes.append(line)

    # Model
    model = models.resnet152(pretrained=True)
    model.to(device)
    model.eval()

    # Image
    raw_image = cv2.imread(image_path)[..., ::-1]
    raw_image = cv2.resize(raw_image, (224,) * 2)
    image = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )(raw_image).unsqueeze(0)
    image = image.to(device)

    gcam = GradCAM(model=model)
    predictions = gcam.forward(image)
    top_idx = predictions[0][1]

    for target_layer in ["layer1", "layer2", "layer3", "layer4"]:
        print("Generating Grad-CAM @{}".format(target_layer))

        # Grad-CAM
        gcam.backward(idx=top_idx)
        region = gcam.generate(target_layer=target_layer)

        save_gradcam(
            "results/{}-gradcam-{}-{}.png".format(
                "resnet152", target_layer, classes[top_idx]
            ),
            region,
            raw_image,
        )


if __name__ == "__main__":
    main()
