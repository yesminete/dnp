#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 16:08:44 2020

@author: reisertm
"""
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
import improc_utils
import customLayers


#%%




trainset = []
labelset = []


img = nib.load('example2d.nii.gz')
a = np.expand_dims(np.expand_dims(np.squeeze(img.get_fdata()),0),3)
trainset.append( tf.convert_to_tensor(a,dtype=tf.float32) )

label = nib.load('example2d_label.nii.gz')
a = np.expand_dims(np.squeeze(label.get_fdata()),0)

labelset.append( tf.convert_to_tensor(a[...,8:9],dtype=tf.float32) )

resolutions = [[1,1]]
 

#%% 2D
nD=2
cgen = patchwork.CropGenerator(
                  scheme = {
                      "destvox_mm": [3,3],
                      "destvox_rel": None,
                      "fov_mm":[100]*nD,
                      "fov_rel":None,
                      "patch_size":[32,32]
                      },
                  ndim=nD,
                  interp_type = 'NN',
                  keepAspect=True,
                  scatter_type = 'NN',
                  depth=2)




model = patchwork.PatchWorkModel(cgen,
                      blockCreator= lambda level,outK : customLayers.createUnet_bi(3,outK=outK,nD=nD),
                       num_labels = 1,
                       num_classes = 1           
                      )


s = 1
xx = trainset[0][0:1,:,::s,...]
res = model.apply_full(xx,resolution=[1,s],
                       generate_type='random',jitter=0.1,   repetitions=100,dphi=0,verbose=True,scale_to_original=False,testIT=True)


print(res.shape)
plt.imshow(tf.squeeze(res),aspect=1/s)
plt.pause(1)
plt.imshow(tf.squeeze(xx),aspect=1/s)
