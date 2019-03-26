# deepfree
keras-style deep network package for classification and prediction
# install
``` python 
pip install --upgrade numpy h5py
pip install --upgrade deepfree
```
# feature
## rapid modeling
established model like ``python DBN``, ``python SAE`` for fast learning.
## stacking blocks
building model like stack blocks via using ``python Model.add_layer([list of Layer])``
a set of ``python Layer`` can be selected, such as ``python PHVariable``, ``python Dense``, ``python MaxPooling2D``,``python Flatten``,``python Concatenate``, ``python MultipleInput``, ``python Conv2D``.
## flexible input
you can set the model's parameters of listed in ``python base/_attribute.py`` when first building the model (``python DBN(para=1)``, ``python SAE(para=1)``, ``python Model(para=1)``) or training (``python Model.training(para=1)``).
# example
a simple DNN can be structed as blow:
```python 
    self.input = PHVariable(self.struct[0])('input')
    self.label = PHVariable(self.struct[-1])('label')
        
    for i in range(len(self.struct)-2):
        self.add_layer(Dense(self.struct[i+1], 
                             activation = self.next_activation(), 
                             is_dropout = True))
    self.add_layer(Dense(self.struct[-1], activation = self.output_func))
 ```
# blog
[Github](https://github.com/fuzimaoxinan/deepfree),
[zhihu](https://www.zhihu.com/people/fu-zi-36-41/posts),
[CSDN](https://blog.csdn.net/fuzimango/article/list/),
QQ Group:640571839 
