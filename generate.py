#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-05-18

from __future__ import print_function

import argparse

import cv2
import numpy as np

import torchvision
from gradcam import BackProp, GradCAM, GuidedBackProp
from torchvision import transforms

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Grad-CAM visualization')
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--image', type=str, required=True)
    args = parser.parse_args()

    # Load the synset words
    file_name = 'synset_words.txt'
    classes = list()
    with open(file_name) as class_file:
        for line in class_file:
            classes.append(line.strip().split(' ', 1)[1].split(', ', 1)[0].replace(' ', '_'))
    classes = tuple(classes)

    print('Loading a model...')
    model = torchvision.models.vgg19(pretrained=True)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    print('\nGrad-CAM')
    gcam = GradCAM(model=model, target_layer='features.36',
                   n_class=1000, cuda=args.cuda)
    gcam.load_image(args.image, transform)
    gcam.forward()

    for i in range(0, 5):
        print('{:.5f}\t{}'.format(gcam.prob[i], classes[gcam.idx[i]]))
        gcam.backward(idx=gcam.idx[i])
        gcam_data = gcam.generate()
        gcam.save(
            'results/gcam_{}.png'.format(classes[gcam.idx[i]]), gcam_data)

    print('\nBackpropagation')
    bp = BackProp(model=model, target_layer='features.0',
                  n_class=1000, cuda=args.cuda)
    bp.load_image(args.image, transform)
    bp.forward()

    for i in range(0, 5):
        print('{:.5f}\t{}'.format(bp.prob[i], classes[bp.idx[i]]))
        bp.backward(idx=bp.idx[i])
        bp_data = bp.generate()
        bp.save('results/bp_{}.png'.format(classes[bp.idx[i]]), bp_data)

    print('\nGuided Backpropagation')
    gbp = GuidedBackProp(model=model, target_layer='features.0',
                         n_class=1000, cuda=args.cuda)
    gbp.load_image(args.image, transform)
    gbp.forward()

    for i in range(0, 5):
        print('{:.5f}\t{}'.format(gbp.prob[i], classes[gbp.idx[i]]))
        gcam.backward(idx=gcam.idx[i])
        gcam_data = gcam.generate()

        gbp.backward(idx=gbp.idx[i])
        gbp_data = gbp.generate()
        gbp.save(
            'results/gbp_{}.png'.format(classes[gbp.idx[i]]), gbp_data.copy())

        gcam_data = gcam_data - np.min(gcam_data)
        gcam_data = gcam_data / np.max(gcam_data)
        gcam_data = cv2.resize(gcam_data, (224, 224))
        gcam_data = cv2.cvtColor(gcam_data, cv2.COLOR_GRAY2BGR)

        ggcam_data = gbp_data * gcam_data
        gbp.save(
            'results/ggcam_{}.png'.format(classes[gbp.idx[i]]), ggcam_data)
