# -*- coding: utf-8 -*-

__all__ = ['_evaluate','_layer','_loss','_model','_train']
# deprecated to keep older scripts who import this from breaking
from deepfree.core._evaluate import Evaluate
from deepfree.core._layer import Activation, Layer, PHVariable, MaxPooling2D, Flatten, Concatenate, Dense, MultipleInput, Conv2D
from deepfree.core._loss import Loss
from deepfree.core._model import Model
from deepfree.core._train import Message, Sess, Train
