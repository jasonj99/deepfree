# -*- coding: utf-8 -*-
import sys
sys.path.append("../base")
sys.path.append("../core")
from _model import Model
from _layer import PHVariable,Dense
from _attribute import RBM_DICT,_AE_DICT
from pre_model import PreModel

class FTModel(Model):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        super(FTModel, self).__init__(**kwargs)
        
    def build_model(self):
        """
        Fine-tuning
        """
        # variable
        self.input = PHVariable(self.struct[0])('input')
        self.label = PHVariable(self.struct[-1])('label')
        
        for i in range(len(self.struct)-2):
            self.add_layer(Dense(self.struct[i+1], 
                                 activation = self.next_activation(), 
                                 is_dropout = True))
        self.add_layer(Dense(self.struct[-1], activation = self.output_func))
        
        if self.is_pre: self.build_pre_model()
        
    def build_pre_model(self):
        """
        Pre-training
        """
        self.kwargs.update({'name':self.pre_name, 
                            'struct':self.struct[:-1], 
                            'layer_list': self.layer_list,
                            'sub_type':self.sub_type})
        self.pre_model = PreModel(**self.kwargs)

class DBN(FTModel):
    def __init__(self, **kwargs):
        default_dict = {'name':'DBN',
                        'pre_name':'DBM',
                        'sub_type':'rbm'
                        }
        default_dict.update(**kwargs)
        kwargs = dict(RBM_DICT, **default_dict)
        super(DBN, self).__init__(**kwargs)

class SAE(FTModel):
    def __init__(self, **kwargs):
        default_dict = {'name':'SAE',
                        'pre_name':'DAE',
                        'sub_type':'ae'
                        }
        default_dict.update(**kwargs)
        kwargs = dict(_AE_DICT, **default_dict)
        super(SAE, self).__init__(**kwargs)