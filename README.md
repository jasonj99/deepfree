# deepfree
Keras-style deep network package for classification and prediction
# install
``` python 
pip install --upgrade numpy h5py
pip install --upgrade deepfree
```
# feature
## rapid modeling
Established model like ``python DBN``, ``python SAE`` for fast learning.
## stacking blocks
Building model like stack blocks via using ``python Model.add_layer([list of Layer])``.<br />
A set of ``python Layer`` can be selected, such as ``python PHVariable``, ``python Dense``, ``python MaxPooling2D``,``python Flatten``,``python Concatenate``, ``python MultipleInput``, ``python Conv2D``.
## flexible setting
You can set the model's parameters of listed in ``python base/_attribute.py`` when first building the model (``python DBN(para=1)``, ``python SAE(para=1)``, ``python Model(para=1)``) or training (``python Model.training(para=1)``). If you do not set a value, the default value in ``python base/_attribute.py`` will be used.
# example
A simple example for constructing and training DNN:
```python
from deepfree import Model
model = Model()
model.struct = [784, 100 ,10]
model.input = PHVariable(self.struct[0])('input')
model.label = PHVariable(self.struct[-1])('label')
        
for i in range(len(model.struct)-2):
    model.add_layer(Dense(model.struct[i+1], 
                         activation = self.next_activation(), 
                         is_dropout = True))
model.add_layer(Dense(model.struct[-1], activation = model.output_func))
model.training(dataset = ... or data_path = ...)
 ```
# blog
[Github](https://github.com/fuzimaoxinan/deepfree),
[zhihu](https://www.zhihu.com/people/fu-zi-36-41/posts),
[CSDN](https://blog.csdn.net/fuzimango/article/list/)<br />
QQ Group:640571839 
