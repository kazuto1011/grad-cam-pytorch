#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-05-18

from __future__ import print_function

import click
import cv2
import numpy as np
import torch
from torch.autograd import Variable

from grad_cam import BackPropagation, GradCAM, GuidedBackPropagation
from torchvision import models, transforms

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


@click.command()
@click.option('--image-path', type=str, required=True)
@click.option('--arch', type=click.Choice(model_names), required=True)
@click.option('--topk', type=int, default=3)
@click.option('--cuda/--no-cuda', default=True)
def main(image_path, arch, topk, cuda):

    CONFIG = {
        'resnet152': {
            'target_layer': 'layer4.2',
            'input_size': 224
        },
        'vgg19': {
            'target_layer': 'features.35',
            'input_size': 224
        },
        'inception_v3': {
            'target_layer': 'Mixed_7c',
            'input_size': 299
        },
    }.get(arch)

    cuda = cuda and torch.cuda.is_available()

    # Synset words
    classes = list()
    with open('samples/synset_words.txt') as lines:
        for line in lines:
            line = line.strip().split(' ', 1)[1]
            line = line.split(', ', 1)[0].replace(' ', '_')
            classes.append(line)

    # Model
    model = models.__dict__[arch](pretrained=True)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Image
    raw_image = cv2.imread(image_path)[:, :, ::-1]
    raw_image = cv2.resize(raw_image, (CONFIG['input_size'], ) * 2)
    image = transform(raw_image).unsqueeze(0)
    image = Variable(image, volatile=False, requires_grad=True)

    if cuda:
        model.cuda()
        image = image.cuda()

    print('1. Grad-CAM')
    gcam = GradCAM(model=model)
    probs, idx = gcam.forward(image)

    for i in range(0, topk):
        gcam.backward(idx=idx[i])
        output = gcam.generate(target_layer=CONFIG['target_layer'])
        gcam.save('results/{}_gcam_{}.png'.format(classes[idx[i]], arch), output, raw_image)  # NOQA
        print('\t{:.5f}\t{}'.format(probs[i], classes[idx[i]]))

    print('2. Vanilla Backpropagation')
    bp = BackPropagation(model=model)
    probs, idx = bp.forward(image)

    for i in range(0, topk):
        bp.backward(idx=idx[i])
        output = bp.generate()
        bp.save('results/{}_bp_{}.png'.format(classes[idx[i]], arch), output)
        print('\t{:.5f}\t{}'.format(probs[i], classes[idx[i]]))

    print('3. Guided Backpropagation/Grad-CAM')
    gbp = GuidedBackPropagation(model=model)
    probs, idx = gbp.forward(image)

    for i in range(0, topk):
        gcam.backward(idx=idx[i])
        region = gcam.generate(target_layer=CONFIG['target_layer'])

        gbp.backward(idx=idx[i])
        feature = gbp.generate()

        output = feature * region[:, :, np.newaxis]
        gbp.save('results/{}_gbp_{}.png'.format(classes[idx[i]], arch), feature)  # NOQA
        gbp.save('results/{}_ggcam_{}.png'.format(classes[idx[i]], arch), output)  # NOQA
        print('\t{:.5f}\t{}'.format(probs[i], classes[idx[i]]))


if __name__ == '__main__':
    main()
