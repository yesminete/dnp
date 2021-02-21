#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys 
sys.path.append("/home/reisertm")
sys.path.append("/home/reisertm/nora/src/python")


import numpy as np
import tensorflow as tf
import math
from tensorflow.keras import Model
from tensorflow.keras import layers


from timeit import default_timer as timer

import nibabel as nib
import matplotlib.pyplot as plt

import patchwork2.model as patchwork
from patchwork2.improc_utils import *

from DPX_core import *
from DPX_improc import *
from patchwork2 import *


#%%

trainset,labelset,resolutions,subjs = load_data_for_training(
                                project = 'tschaga',    
                                subjects = '#Tag:dmritest',
                               # subjects = '15341572', #'#Tag:MIDItrain',
                                contrasts_selector = ['dmri_sms.nii'],
                               # labels_selector = ['mask_untitled0.nii.gz','mask_untitled0.nii.gz'],
                               # one_hot_index_list=[0,1],
                                labels_selector = ['dmri_sms.nii'],
                                threshold=None,
                              #  labels_selector = ['forms/untitledrrr.form.json'],
                                add_inverted_label=False,
                               # reslice='2mm,3mm,3mm',
                                max_num_data=1
                                )

#%%
nD=3
cgen = patchwork.CropGenerator(
                  scheme = { 
                      "destvox_mm": [1.5]*nD,
                      "destvox_rel": None,
                      "fov_mm":[12]*nD,
                      "fov_rel":None,
                      "patch_size":[(8,8,8),(32,32,32)],
                      "out_patch_size": [(32,32,32),(8,8,8)]
                      },
                  ndim=nD,
                  transforms=['tensor']*2,
                  depth=2)
def thenet(level):
    verbose = False;
    if level == 0:
        return customLayers.createTnet(input_shape=[8,8,8],fdims=[32,32,16],
                                                           direction=0,verbose=verbose,out=1)
    else:
        return customLayers.createTnet(input_shape=[32,32,32],fdims=[None,10,12],
                                                           direction=1,verbose=verbose,out=6)

        

model = patchwork.PatchWorkModel(cgen,
                      blockCreator= lambda level,outK : thenet(level),
                      forward_type='noinput',
                      intermediate_loss=True,
                      block_out=[1,6],
                      num_labels = -1
                      )


#%%

res = model.apply_full(trainset[0][0:1,...],resolution=resolutions[0], repetitions=50,verbose=True,
                       scale_to_original=False,level=0)

plt.imshow(res[:,:,100,0])

#%%

model.modelname = "models/test2"

model.train(trainset,labelset,
            resolutions=resolutions,
            fit_type='custom',
            valid_ids = [],
            train_ids = [0],
            num_its=1,
            debug=True,
            loss = [None,tf.keras.losses.MeanSquaredError()],
            augment={'dphi':0.1},
            num_patches=30,
            epochs=10)

#%%

a = tf.ones([1,8,8,8,6]); x = customLayers.createTnet(input_shape=[8,8,8],fdims=[32,32,16],
                                                           direction=0,verbose=True)

print(x(a).shape)
#%%
a = tf.ones([1,32,32,32,1]); x = customLayers.createTnet(input_shape=[32,32,32],fdims=[None,10,12],
                                                           direction=1,verbose=True,out=6)

print(x(a).shape)
