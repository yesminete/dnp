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

import matplotlib.pyplot as plt

import patchwork2.model as patchwork
import patchwork2.improc_utils
import patchwork2.customLayers as customLayers


#%%

fi = 't1.nii'
fi = '/software/patchwork2/s004.nii'

trainset = []
labelset = []

img = np.random.rand(32, 32, 32).astype(np.float32)
ie = np.eye(4)

#ie = np.array([[-1,0,0,100],[0,-1,0,-10000],[0,0,-1,-500],[0,0,0,1]])
if 0:
    ie = np.array([ [0,0,-0.6,100],
                    [-0.6,0,0,-10000],
                    [0,-1,0,-500],
                    [0,0,0,1]])
#img = img[:,:,::2]
img = img[:,:,:]


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
                  scheme = {
                      #"destvox_mm": [1,1,1],
                      "destvox_rel": [2,2,2],
                      #"fov_mm":[10,10,10],
                      "fov_rel":[0.8,0.8,0.8],
                      "patch_size":[32,32,32]
                      },
                  ndim=nD,
                  interp_type = 'NN',
                  scatter_type = 'NN',
                  system='world',
                  depth=2)




model = patchwork.PatchWorkModel(cgen,
                      blockCreator= lambda level,outK : customLayers.createUnet_v2(2,1,nD=3,verbose=False),
                      intermediate_loss = True,
                       num_labels = 5,
                       num_classes = 1,                      
                      )
#%%


s = 1
xx = trainset[0]
res = model.apply_full(xx,resolution=resolutions[0],
                       #branch_factor=1,
                       #level='mix',
                       #lazyEval=0.1,
                       generate_type='random_fillholes',
                       #generate_type='random',
                       jitter=0,   repetitions=64,num_chunks=4,
                       augment= {"independent_augmentation" : False,
                                 "dphi" : 0 },
                       verbose=True,scale_to_original=False,testIT=1)



s=1
#plt.imshow(tf.squeeze(res[150,:,:]),aspect=1/s)
plt.imshow(tf.squeeze(res[:,:,20])>0.0,aspect=1/s)
print(res.shape)
#%%

res,nii = model.apply_on_nifti(fi, ['/nfs/noraimg/transfer/xx.nii'],
                       branch_factor=1,
                       #level='mix',
                       generate_type='random_fillholes',jitter=0,   repetitions=20,num_chunks=4,
                       augment= {"independent_augmentation" : False,
                                 "dphi" : 0.4 },
                       verbose=True,scale_to_original=False,testIT=1)
















