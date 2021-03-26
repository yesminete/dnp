#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 13 14:08:36 2021

@author: reisertm
"""

from .customLayers import *

numinput = [4, 1, 1, 1, 2, 1, 1, 1, 1, 1]
onehots = [[1,2,3], [1], [1,2], [1,2], [1,2], [1], [1,2], [1,2], [1], [1] ]
vsizes = [[1.00, 1.00, 1.00], 
         [1.25, 1.25, 1.37], 
         [0.77, 0.77, 1.00], 
         [1.00, 1.00, 1.00], 
         [0.62, 0.62, 3.60], 
         [0.79, 0.79, 1.24], 
         [0.80, 0.80, 2.50], 
         [0.80, 0.80, 5.00], 
         [0.79, 0.79, 5.00], 
         [0.78, 0.78, 5.00] ]
aniso = [0,0,0,0,1,0,1,1,1,1]
wid80 =[[96.0, 136.0, 104.0 ],
        [320.00, 320.00, 126.04 ],
        [314.40, 314.40, 345.60 ],
        [28.00, 40.00, 28.80 ],
        [160.00, 160.00, 57.60 ],
        [321.60, 321.60, 250.99 ],
        [328.80, 328.80, 186.00 ],
        [327.20, 327.20, 196.00 ],
        [324.80, 324.80, 360.00 ],
        [320.00, 320.00, 380.00 ]]
ct = [0,0,1,0,0, 1,1,1,1,1]
flips = [[1,0,0],  [1,0,0],  [1,0,0],  [0,0,0],  [1,0,0],
         [1,0,0],  [1,0,0],  [1,0,0],  [1,0,0],  [1,0,0] ]

theblock = lambda level,outK,input_shape : createUnet_v2(depth=5,
                 outK=outK,nD=3,input_shape=input_shape,feature_dim=[32,32,32,32,32],dropout=False)
#theblock = lambda level,outK,input_shape : createUnet_v2(depth=5,
 #                outK=outK,nD=3,input_shape=input_shape,feature_dim=[8,16,16,32,64],dropout=False)


bala={"ratio":0.5}
schemeP = lambda task : { 
    "destvox_mm": vsizes[task],
    "destvox_rel": None,
    "fov_mm":wid80[task],
    "fov_rel":None,
}
depth=5
ident=False
fittyp = 'custom'

#myloss = TopK_loss3D(K=1000,combi=True)


import tensorflow_addons as tfa

l = tfa.losses.SigmoidFocalCrossEntropy(from_logits=True)
t = tfa.losses.SigmoidFocalCrossEntropy(from_logits=False)

#myloss = TopK_loss3D(K=1000,combi=True)
myloss = [l,l,l,l,t]

#myloss = None
myoptim = None #tfa.optimizers.AdamW()
#myoptim = tf.optimizers.Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=True)





def augmentP(task):

    if aniso[task] == 1:
        return { 'dphi': [0.1,0,0], 'flip': flips[task], 'dscale':[0.1,0.1,0.1] }
    else:
  #     return { 'dphi': [0.1,0.1,0.1], 'flip': flips[task], 'dscale':[0.1,0.1,0.1] }
        return { 'dphi': 0, 'flip': 0, 'dscale':0 }

def preprocP(task):
    
    if ct[task]==1:
         preproc = lambda level: HistoMaker(nD=3,init='ct')
         normtyp = None
    else:
         preproc = None #lambda level: HistoMaker(nD=3,out=10,init=None)
         normtyp = 'mean'
         
    return preproc,normtyp

















