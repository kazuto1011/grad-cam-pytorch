#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-05-18


import sys
from collections import OrderedDict

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

import torchvision
from torchvision import transforms

cuda = True


class GradCAM(object):

    def __init__(self, model, target_layer):
        self.model = model
        if cuda:
            self.model.cuda()
        self.model.eval()
        self.target_layer = target_layer
        self.outputs_forward = OrderedDict()
        self.outputs_backward = OrderedDict()
        self._set_hook_func()

    def _func_f(self, m, i, o):
        self.outputs_forward[id(m)] = o.data.cpu()

    def _func_b(self, m, i, o):
        self.outputs_backward[id(m)] = o[0].cpu()

    def _set_hook_func(self):
        for module in self.model.named_modules():
            module[1].register_forward_hook(self._func_f)
            module[1].register_backward_hook(self._func_b)

    def _one_hot(self, idx):
        one_hot = torch.FloatTensor(1, 1000)
        one_hot.zero_()
        one_hot[0][idx] = 1.0
        return one_hot

    def _normalize(self, gradients):
        l2_norm = torch.sqrt(torch.mean(torch.pow(gradients, 2))) + 1e-5
        return gradients / l2_norm.data[0]

    def _retrieve_conv_outputs(self, outputs, target_layer):
        for key, value in outputs.items():
            for module in model.named_modules():
                if id(module[1]) == key:
                    if module[0] == target_layer:
                        return value

    def load_image(self, filename, transform):
        self.image = cv2.imread(sys.argv[1])[:, :, ::-1]
        self.image = cv2.resize(self.image, (224, 224))
        self.image_ = transform(self.image).unsqueeze(0)
        if cuda:
            self.image_ = self.image_.cuda()
        self.image_ = Variable(self.image_, volatile=False)

    def forward(self):
        self.output = self.model.forward(self.image_)
        self.output = F.softmax(self.output)
        self.prob, self.idx = self.output.data.squeeze().sort(0, True)

    def backward(self, idx):
        classwise = self._one_hot(idx)
        if cuda:
            classwise = classwise.cuda()
        self.output.backward(classwise, retain_variables=True)

        self.featuremaps = self._retrieve_conv_outputs(
            self.outputs_forward, self.target_layer)
        self.gradients = self._retrieve_conv_outputs(
            self.outputs_backward, self.target_layer)

    def compute_gradient_weights(self):
        self.gradients = self._normalize(self.gradients)
        self.map_size = self.gradients.size()[2:]
        self.weights = nn.AvgPool2d(self.map_size)(self.gradients)

    def computing_cam(self):
        grad_cam = torch.FloatTensor(self.map_size).zero_()
        for fmap, weight in zip(self.featuremaps[0], self.weights[0]):
            grad_cam += fmap * weight.data[0][0]
        return F.relu(Variable(grad_cam))

    def save(self, filename, grad_cam):
        grad_cam = grad_cam.data.numpy()
        grad_cam = grad_cam - np.min(grad_cam)
        grad_cam = grad_cam / np.max(grad_cam)

        grad_cam = cv2.resize(grad_cam, (224, 224))
        grad_cam = cv2.applyColorMap(
            np.uint8(255 * grad_cam), cv2.COLORMAP_JET)
        grad_cam = np.asarray(grad_cam, dtype=np.float) + \
            np.asarray(self.image, dtype=np.float)
        grad_cam = 255 * grad_cam / np.max(grad_cam)
        grad_cam = np.uint8(grad_cam)

        cv2.imwrite(filename, grad_cam)


file_name = 'synset_words.txt'
classes = list()
with open(file_name) as class_file:
    for line in class_file:
        classes.append(line.strip().split(' ', 1)[1].split(', ', 1)[0])
classes = tuple(classes)


if __name__ == '__main__':
    model = torchvision.models.vgg19(pretrained=True)
    target_layer = 'features.36'
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225])
    ])

    cam = GradCAM(model, target_layer)
    cam.load_image(sys.argv[1], transform)
    cam.forward()
    for i in range(0, 5):
        print '{:.5f}\t{}'.format(cam.prob[i], classes[cam.idx[i]])
        cam.backward(cam.idx[i])
        cam.compute_gradient_weights()
        grad_cam = cam.computing_cam()
        cam.save(
            'results/grad_cam_{}.png'.format(classes[cam.idx[i]]), grad_cam)
