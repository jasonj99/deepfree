# -*- coding: utf-8 -*-

<<<<<<< HEAD
__all__ = ['_evaluate','_layer','_loss','_model','_submodel','_train']
=======
__all__ = ['_evaluate','_layer','_loss','_model','_train']
>>>>>>> 987acc1d5a935b80c5ee1c424ca93f2b580c8c7f
# deprecated to keep older scripts who import this from breaking
from deepfree.core._evaluate import Evaluate
from deepfree.core._layer import Activation, phvariable, maxpooling2d, flatten, concatenate, noise, Layer, Dense, Conv2D
from deepfree.core._loss import Loss
from deepfree.core._model import Model
<<<<<<< HEAD
from deepfree.core._submodel import SubModel
=======
>>>>>>> 987acc1d5a935b80c5ee1c424ca93f2b580c8c7f
from deepfree.core._train import Message, Sess, Train
