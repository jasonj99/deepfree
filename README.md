# deepfree
Keras-style deep network package for classification and prediction
# install
``` python 
pip install --upgrade numpy h5py
pip install --upgrade deepfree
```
# feature
## rapid modeling
Established model like ``DBN``, ``SAE`` for fast learning.
## stacking blocks
Building model like stack blocks via using ``Model.add_layer([list of Layer])``.<br />
A set of ``Layer`` can be selected, such as ``PHVariable``, ``Dense``, ``MaxPooling2D``,``Flatten``,``Concatenate``, ``MultipleInput``, ``Conv2D``.
## flexible setting
You can set the model's parameters of listed in ``python base/_attribute.py`` when first building the model (``DBN(para=...)``, ``SAE(para=...)``, ``Model(para=...)``) or training (``Model.training(para=...)``). If you do not set a value, the default value in ``base/_attribute.py`` will be used.
# example
Simple construction and training for DNN as shown below:
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
