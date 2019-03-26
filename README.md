# deepfree
Keras-style deep network package for classification and prediction
# install
``` python 
pip install --upgrade numpy h5py
pip install --upgrade deepfree
```
# feature
## fast learning
The main framework of the program relies on ``Model`` in ``core._model`` and ``Layer`` in ``core._layer``, which can import directly through ``'from deepfree import Model, Layer'``. You can quickly build and train the model by using them flexibly.<br />
In addition, the constructed ``DBN`` and ``SAE`` can be employed directly, which are inherited from ``Model``.
## stacking blocks
By calling ``Model.add_layer(['a Layer of a list of Layer'])``, you can build the model like stack the blocks .<br />
There are a set of ``Layer`` can be selected, such as ``PHVariable``, ``Dense``, ``MaxPooling2D``,``Flatten``,``Concatenate``, ``MultipleInput``, ``Conv2D``.
## flexible setting
You can set the model's parameters listed in ``base._attribute`` when first building model (``DBN(para=...)``, ``SAE(para=...)``, ``Model(para=...)``) or training it (``Model.training(para=...)``). If you do not set a value, the default value in ``base._attribute`` will be applied.
# example
A simple DNN can be constructed and trained as:
```python
from deepfree import Model
from deepfree import PHVariable,Dense
model = Model()
model.struct = [784, 100 ,10]
model.input = PHVariable(model.struct[0])('input')
model.label = PHVariable(model.struct[-1])('label')
        
for i in range(len(model.struct)-2):
    model.add_layer(Dense(model.struct[i+1], 
                         activation = model.next_activation(), 
                         is_dropout = True))
model.add_layer(Dense(model.struct[-1], activation = model.output_func))
model.training(dataset = ...,data_path = ...)
```
# blog
[Github](https://github.com/fuzimaoxinan/deepfree),
[zhihu](https://www.zhihu.com/people/fu-zi-36-41/posts),
[CSDN](https://blog.csdn.net/fuzimango/article/list/)<br />
QQ Group:640571839 
