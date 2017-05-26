#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-05-26

from __future__ import print_function

from collections import OrderedDict

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F


class PropBase(object):

    def __init__(self, model, target_layer, n_class, cuda=True):
        self.model = model
        self.cuda = cuda
        if self.cuda:
            self.model.cuda()
        self.model.eval()
        self.target_layer = target_layer
        self.n_class = n_class
        self.probs = None
        self.outputs_forward = OrderedDict()
        self.outputs_backward = OrderedDict()
        self.set_hook_func()

    def set_hook_func(self):
        raise NotImplementedError

    def encode_one_hot(self, idx):
        one_hot = torch.FloatTensor(1, self.n_class).zero_()
        one_hot[0][idx] = 1.0
        return one_hot

    def load_image(self, filename, transform):
        self.image = cv2.imread(filename)[:, :, ::-1]
        self.image = cv2.resize(self.image, (224, 224))
        self.image_ = transform(self.image).unsqueeze(0)
        if self.cuda:
            self.image_ = self.image_.cuda()
        self.image_ = Variable(self.image_, volatile=False, requires_grad=True)

    def forward(self):
        self.probs = F.softmax(self.model.forward(self.image_))
        self.prob, self.idx = self.probs.data.squeeze().sort(0, True)

    def backward(self, idx):
        self.model.zero_grad()
        one_hot = self.encode_one_hot(idx)
        if self.cuda:
            one_hot = one_hot.cuda()
        self.probs.backward(one_hot, retain_variables=True)

    def get_conv_outputs(self, outputs, target_layer):
        for key, value in outputs.items():
            for module in self.model.named_modules():
                if id(module[1]) == key:
                    if module[0] == target_layer:
                        return value
        raise ValueError('invalid layer name: {}'.format(target_layer))


class GradCAM(PropBase):

    def set_hook_func(self):

        def func_f(module, input, output):
            self.outputs_forward[id(module)] = output.data.cpu()

        def func_b(module, grad_in, grad_out):
            self.outputs_backward[id(module)] = grad_out[0].cpu()

        for module in self.model.named_modules():
            module[1].register_forward_hook(func_f)
            module[1].register_backward_hook(func_b)

    def normalize(self, grads):
        l2_norm = torch.sqrt(torch.mean(torch.pow(grads, 2))) + 1e-5
        return grads / l2_norm.data[0]

    def compute_gradient_weights(self):
        self.grads = self.normalize(self.grads)
        self.map_size = self.grads.size()[2:]
        self.weights = nn.AvgPool2d(self.map_size)(self.grads)

    def generate(self):
        self.fmaps = self.get_conv_outputs(
            self.outputs_forward, self.target_layer)
        self.grads = self.get_conv_outputs(
            self.outputs_backward, self.target_layer)
        self.compute_gradient_weights()

        gcam = torch.FloatTensor(self.map_size).zero_()
        for fmap, weight in zip(self.fmaps[0], self.weights[0]):
            gcam += fmap * weight.data[0][0]

        gcam = F.relu(Variable(gcam))

        return gcam.data.numpy()

    def save(self, filename, gcam):
        gcam = gcam - np.min(gcam)
        gcam = gcam / np.max(gcam)

        gcam = cv2.resize(gcam, (224, 224))
        gcam = cv2.applyColorMap(
            np.uint8(255 * gcam), cv2.COLORMAP_JET)
        gcam = np.asarray(gcam, dtype=np.float) + \
            np.asarray(self.image, dtype=np.float)
        gcam = 255 * gcam / np.max(gcam)
        gcam = np.uint8(gcam)

        cv2.imwrite(filename, gcam)


class BackProp(PropBase):

    def set_hook_func(self):

        def func_b(module, grad_in, grad_out):
            self.outputs_backward[id(module)] = grad_in[0].cpu()

        for module in self.model.named_modules():
            module[1].register_backward_hook(func_b)

    def generate(self):
        self.output = self.get_conv_outputs(
            self.outputs_backward, self.target_layer)
        return self.output.data.numpy()[0].transpose(1, 2, 0)

    def save(self, filename, data):
        data /= np.maximum(-1 * data.min(), data.max())
        data *= 128
        data += 128
        cv2.imwrite(filename, data)


class GuidedBackProp(BackProp):

    def set_hook_func(self):

        def func_b(module, grad_in, grad_out):
            self.outputs_backward[id(module)] = grad_in[0].cpu()

            # Cut off negative gradients
            if isinstance(module, nn.ReLU):
                return (F.threshold(grad_in[0], threshold=0.0, value=0.0),)

        for module in self.model.named_modules():
            module[1].register_backward_hook(func_b)
