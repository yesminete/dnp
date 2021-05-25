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

img = nib.load('t1.nii')
ie = img.affine
img = img.get_fdata()

#ie = np.array([[-1,0,0,100],[0,-1,0,-10000],[0,0,-1,-500],[0,0,0,1]])
ie = np.array([[-3,0,0,100],[0,-1,0,-10000],[0,0,1,-500],[0,0,0,1]])
img = img[:,:,::2]

img = tf.cast(tf.expand_dims(tf.expand_dims(img,0),4),dtype=tf.float32)
resolutions = [{"input_edges":ie}]



#img = tf.ones([1,60, 60,30,1]);
#resolutions = [{"input_edges":ie}]
#resolutions = [[1,1,1.5]]



trainset = [img]

#% 3D
#tf.random.set_seed(2)
nD=3
cgen = patchwork.CropGenerator(
    snapper = [0,1,1],
                  scheme = {
                      #"destvox_mm": [1,1,1],
                      "destvox_rel": [5,1,1],
                      "fov_mm":[100,100,150],
                      #"fov_rel":[0.5,0.5,0.5],
                      "patch_size":[32,32,32]
                      },
                  ndim=nD,
                  interp_type = 'NN',
                  scatter_type = 'NN',
                  depth=3)




model = patchwork.PatchWorkModel(cgen,
                      blockCreator= lambda level,outK : customLayers.createUnet_v2(2,1,nD=3,verbose=False),
                      intermediate_loss = True,
                       num_labels = 1,
                       num_classes = 1,                      
                      )
#%


s = 1
xx = trainset[0]
res = model.apply_full(xx,resolution=resolutions[0],
                       branch_factor=1,
                       level='mix',
                       generate_type='random',jitter=0.1,   repetitions=10,
                       augment= {"independent_augmentation" : False,
                                 "dphi" : [0,0 ,0] },
                       verbose=True,scale_to_original=False,testIT=1)



s=2
plt.imshow(tf.squeeze(res[20,:,:]),aspect=1/s)

