#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-05-18

from __future__ import print_function

import sys
from collections import OrderedDict

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import transforms

cuda = True


class Base(object):

    def __init__(self, model, target_layer, n_class):
        self.model = model
        if cuda:
            self.model.cuda()
        self.model.eval()
        self.target_layer = target_layer
        self.n_class = n_class
        self._output = None
        self._outputs_forward = OrderedDict()
        self._outputs_backward = OrderedDict()
        self._set_hook_func()

    def _set_hook_func(self):
        raise NotImplementedError

    def _one_hot(self, idx):
        one_hot = torch.FloatTensor(1, self.n_class).zero_()
        one_hot[0][idx] = 1.0
        return one_hot

    def load_image(self, filename, transform):
        self.image = cv2.imread(sys.argv[1])[:, :, ::-1]
        self.image = cv2.resize(self.image, (224, 224))
        self.image_ = transform(self.image).unsqueeze(0)
        if cuda:
            self.image_ = self.image_.cuda()
        self.image_ = Variable(self.image_, volatile=False, requires_grad=True)

    def forward(self):
        self._output = self.model.forward(self.image_)
        self._output = F.softmax(self._output)
        self.prob, self.idx = self._output.data.squeeze().sort(0, True)

    def backward(self):
        raise NotImplementedError

    def _retrieve_conv_outputs(self, outputs, target_layer):
        for key, value in outputs.items():
            for module in self.model.named_modules():
                if id(module[1]) == key:
                    if module[0] == target_layer:
                        return value
        raise ValueError('invalid layer name: {}'.format(target_layer))


class GradCAM(Base):

    def _set_hook_func(self):

        def _func_f(module, input, output):
            self._outputs_forward[id(module)] = output.data.cpu()

        def _func_b(module, grad_in, grad_out):
            self._outputs_backward[id(module)] = grad_out[0].cpu()

        for module in self.model.named_modules():
            module[1].register_forward_hook(_func_f)
            module[1].register_backward_hook(_func_b)

    def _normalize(self, grads):
        l2_norm = torch.sqrt(torch.mean(torch.pow(grads, 2))) + 1e-5
        return grads / l2_norm.data[0]

    def _compute_gradient_weights(self):
        self.grads = self._normalize(self.grads)
        self.map_size = self.grads.size()[2:]
        self.weights = nn.AvgPool2d(self.map_size)(self.grads)

    def backward(self, idx):
        classwise = self._one_hot(idx)
        if cuda:
            classwise = classwise.cuda()
        self._output.backward(classwise, retain_variables=True)

    def generate_gcam(self):
        self.fmaps = self._retrieve_conv_outputs(
            self._outputs_forward, self.target_layer)
        self.grads = self._retrieve_conv_outputs(
            self._outputs_backward, self.target_layer)
        self._compute_gradient_weights()

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


class BackProp(Base):

    def _set_hook_func(self):

        def _func_b(module, grad_in, grad_out):
            self._outputs_backward[id(module)] = grad_in[0].cpu()

        for module in self.model.named_modules():
            module[1].register_backward_hook(_func_b)

    def backward(self, idx):
        classwise = self._one_hot(idx)
        if cuda:
            classwise = classwise.cuda()
        self._output.backward(classwise, retain_variables=True)

    def generate(self):
        self.bp = self._retrieve_conv_outputs(
            self._outputs_backward, self.target_layer)
        return self.bp.data.numpy()[0].transpose(1, 2, 0)

    def save(self, filename, data):
        data /= np.maximum(-1 * data.min(), data.max())
        data *= 128
        data += 128
        cv2.imwrite(filename, data)


class GuidedBackProp(Base):

    def _set_hook_func(self):

        def _func_b(module, grad_in, grad_out):
            self._outputs_backward[id(module)] = grad_in[0].cpu()
            if isinstance(module, nn.ReLU):
                return (F.threshold(grad_in[0], threshold=0.0, value=0.0),)

        for module in self.model.named_modules():
            module[1].register_backward_hook(_func_b)

    def backward(self, idx):
        classwise = self._one_hot(idx)
        if cuda:
            classwise = classwise.cuda()
        self._output.backward(classwise, retain_variables=True)

    def generate(self):
        self.gbp = self._retrieve_conv_outputs(
            self._outputs_backward, self.target_layer)
        return self.gbp.data.numpy()[0].transpose(1, 2, 0)

    def save(self, filename, data):
        data /= np.maximum(-1 * data.min(), data.max())
        data *= 128
        data += 128
        cv2.imwrite(filename, data)


if __name__ == '__main__':
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
    gcam = GradCAM(model=model, target_layer='features.36', n_class=1000)
    gcam.load_image(sys.argv[1], transform)
    gcam.forward()

    for i in range(0, 5):
        print('{:.5f}\t{}'.format(gcam.prob[i], classes[gcam.idx[i]]))
        gcam.backward(idx=gcam.idx[i])
        gcam_data = gcam.generate_gcam()
        gcam.save(
            'results/gcam_{}.png'.format(classes[gcam.idx[i]]), gcam_data)

    print('\nBackpropagation')
    bp = BackProp(model=model, target_layer='features.0', n_class=1000)
    bp.load_image(sys.argv[1], transform)
    bp.forward()

    for i in range(0, 5):
        print('{:.5f}\t{}'.format(bp.prob[i], classes[bp.idx[i]]))
        bp.backward(idx=bp.idx[i])
        bp_data = bp.generate()
        bp.save('results/bp_{}.png'.format(classes[bp.idx[i]]), bp_data)

    print('\nGuided Backpropagation')
    gbp = GuidedBackProp(model=model, target_layer='features.0', n_class=1000)
    gbp.load_image(sys.argv[1], transform)
    gbp.forward()

    for i in range(0, 5):
        print('{:.5f}\t{}'.format(gbp.prob[i], classes[gbp.idx[i]]))
        gcam.backward(idx=gcam.idx[i])
        gcam_data = gcam.generate_gcam()

        gbp.backward(idx=gbp.idx[i])
        gbp_data = gbp.generate()
        gbp.save('results/gbp_{}.png'.format(classes[gbp.idx[i]]), gbp_data.copy())

        gcam_data = gcam_data - np.min(gcam_data)
        gcam_data = gcam_data / np.max(gcam_data)
        gcam_data = cv2.resize(gcam_data, (224, 224))
        gcam_data = cv2.cvtColor(gcam_data, cv2.COLOR_GRAY2BGR)

        ggcam_data = gbp_data * gcam_data
        gbp.save('results/ggcam_{}.png'.format(classes[gbp.idx[i]]), ggcam_data)
