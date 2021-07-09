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
import patchwork2.improc_utils
import patchwork2.customLayers as customLayers


#%%




trainset = []
labelset = []



img = nib.load('example2d.nii.gz')
a = np.expand_dims(np.expand_dims(np.squeeze(img.get_fdata()),0),3)
trainset.append( tf.convert_to_tensor(a,dtype=tf.float32) )

label = nib.load('example2d_label.nii.gz')
a = np.expand_dims(np.squeeze(label.get_fdata()),0)

labelset.append( tf.convert_to_tensor(a[...,8:9],dtype=tf.float32) )


#%% 2D
tf.random.set_seed(12)
nD=2
cgen = patchwork.CropGenerator(
                  scheme = {
                      "destvox_mm": [2,4],
                      "destvox_rel": None,
                      #"fov_mm":[100]*nD,
                      "fov_rel":[0.5,0.5],
                      "patch_size":[32,32]
                      },
                  ndim=nD,
                  interp_type = 'NN',
         #         system='world',
                  keepAspect=True,
                  scatter_type = 'NN',
                  depth=1)




model = patchwork.PatchWorkModel(cgen,
                      blockCreator= lambda level,outK : customLayers.createUnet_v2(2,1,nD=2,verbose=True),
                      intermediate_loss = True,
                       num_labels = 1,
                       num_classes = 1       ,
                       
                      )





 
phi = 0.2
s = 2

resolutions = [{"voxsize":[1,1,1],"input_edges":np.array([[math.cos(phi),2*math.sin(phi),0,100],
                                                          [-math.sin(phi),2*math.cos(phi),0,200],
                                                          [0,0,1,-200],
                                                          [0,0,0,1]])}]

xx = trainset[0][0:1,:,::s,...]
rr = [1,s]
rr = resolutions[0]
res = model.apply_full(xx,resolution=rr,
                       branch_factor=1,
                       generate_type='random',jitter=0.1,   repetitions=5,
                       augment= {"independent_augmentation" : True,
                                 "dphi" : 0.0 },
                       verbose=True,scale_to_original=False,testIT=False)

asp = xx.shape[1]/xx.shape[2]/res.shape[0]*res.shape[1]

print(res.shape)
plt.imshow(tf.squeeze(res),aspect=1/s,vmin=0,vmax=500)
#plt.imshow(tf.squeeze(res),aspect=asp)
#plt.pause(1)
#plt.imshow(tf.squeeze(xx),aspect=1/s)
