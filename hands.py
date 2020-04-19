#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 16:08:44 2020

@author: reisertm
"""
import sys 
sys.path.append("/home/reisertm")


import numpy as np
import tensorflow as tf
import math
from tensorflow.keras import Model
from tensorflow.keras import layers

from timeit import default_timer as timer

import nibabel as nib
import matplotlib.pyplot as plt

import patchwork.model as patchwork
from patchwork.improc_utils import *


trainset = []
labelset = []
class_set = []

skip = 1

flip = [0,0,1,1,1]
nfacs = np.zeros([15,5])
for k in range(2):
  img = nib.load('data/hands2/data' + str(k+1) + '.nii.gz')
  a = np.expand_dims(np.expand_dims(np.squeeze(img.get_fdata()),0),3)
  a = a[:,0::skip,0::skip,:]
  if flip[k] == -1:
    a = np.flip(a,1)
    
  #plt.imshow(np.squeeze(a[0,:,:,0]))
  trainset.append( tf.convert_to_tensor(a,dtype=tf.float32) )


  label = nib.load('data/hands2/labels' + str(k+1) + '.nii.gz')
  a = np.expand_dims(np.squeeze(label.get_fdata()),0)
  a = a[:,0::skip,0::skip,:]
  bgnd = np.expand_dims(np.any(a,3)==0,3)
  a = np.concatenate([a,bgnd],3)
  #if flip[k] == -1:
  #  a = np.flip(a,1)
  for j in range(15):
    nfacs[j,k] = math.sqrt(np.sum(a[:,:,:,j]))

  labelset.append( tf.convert_to_tensor(a,dtype=tf.float32)   )
#  labelset.append( [ tf.convert_to_tensor([[flip[k]]]) , a] )
  
 # labelset.append(  tf.convert_to_tensor([[flip[k]]])  )
  #plt.pause(0.001)


#nfacs = np.amax(nfacs,1)
#for k in range(5):
#  for j in range(15):
#      labelset[k][:,:,:,j] = labelset[k][:,:,:,j] / nfacs[j] * 5000

#for k in range(5):
#  labelset[k] =  tf.convert_to_tensor( labelset[k] , dtype=tf.float32)



#%%

################################################################################################




biConvolution = patchwork.biConvolution

def BNrelu(name=None):
  return [layers.BatchNormalization(name=name), layers.LeakyReLU()]

n = 10

def conv_down(name=None,dest=None):
  x = biConvolution(n,3,name=name)
  x.dest = dest
  #return layers.Conv2D(n,3,padding='SAME') 
  return x
def conv_up(name=None):
  #return layers.Conv2DTranspose(n,3,padding='SAME',strides=(2,2)) 
  return biConvolution(n,3,transpose=True,strides=(2,2),name=name)
def conv_out(outK,name=None):
  #return layers.Conv2D(outK,3,padding='SAME') 
  return biConvolution(outK,3,name=name)

def createBlock_(name=None,depth=2,outK=1):
  block = patchwork.CNNblock(name=name)
  for z in range(depth):
    id_d = name+str(1000 + z+1)
    id_u = name+str(2000 + depth-z+1)
    block.add( conv_down(id_d+"1conv")) #  conv_down(name=id_d+"2conv",  dest=id_u+"relu" ) )
    block.add( BNrelu(name=id_d+"2relu") + [layers.MaxPooling2D(pool_size=(2,2)) ] )    
    block.add( conv_up(name=id_u+"conv"))
    block.add( BNrelu(name=id_u+"relu"))
  block.add([layers.Dropout(name=name+"3001",rate=0.5) , conv_out(outK)] )
  return block

def createBlock(depth=2,outK=1):
  theLayers = {}
  for z in range(depth):
    id_d = str(1000 + z+1)
    id_u = str(2000 + depth-z+1)
    theLayers[id_d+"conv"] = [{'f': conv_down() } , {'f': conv_down(), 'dest':id_u+"relu" }  ]
    theLayers[id_d+"relu"] = BNrelu() + [layers.MaxPooling2D(pool_size=(2,2)) ]
    theLayers[id_u+"conv"] =  [conv_up()]
    theLayers[id_u+"relu"] = BNrelu()
  theLayers["3000"] =  [layers.Dropout(rate=0.5), conv_out(outK)]
  return patchwork.CNNblock(theLayers)

def createClassifier(name=None,depth=4,outK=2):
  theLayers = {}
  for z in range(depth):
    id_d = str(1000 + z+1)
    theLayers[id_d+"conv"] = conv_down()
    theLayers[id_d+"relu"] = BNrelu() + [layers.MaxPooling2D(pool_size=(2,2)) ]
  theLayers["3001"] =  layers.Reshape((40,))
  theLayers["3002"] =  layers.Dense(outK)
  theLayers["3003"] = layers.Activation('sigmoid')
    
  return patchwork.CNNblock(theLayers)


#x = createBlock_(name="aa",outK=3)
#print(x(trainset[0][0:1,0:32,0:32,:]).shappatchwork.CNNblock()e)
#y = createBlock(outK=3)
#print(y(trainset[0][0:1,0:32,0:32,:]).shape)
#%%

# x = createBlock()
# tmp = tf.tile(trainset[0],[4,1,1,1])
# tmp = tmp[:,0:64,0:64,:]
# q = x(tmp)
# print(q.shape)
#model = patchwork.PatchWorkModel.load('yyy',custom_objects={'BNrelu':BNrelu})

cgen = patchwork.CropGenerator(patch_size = (16,16), 
                  scale_fac = 0.3, 
                  init_scale = [48,96],
                  depth=2)


model = patchwork.PatchWorkModel(cgen,
                      #blockCreator= lambda level,outK : createBlock_(name='block'+str(level),outK=outK),
                      blockCreator= lambda level,outK : createBlock(outK=outK),
                      preprocCreator = lambda level: patchwork.normalizedConvolution(nD=2),
                    #  classifierCreator = lambda level,outK: createClassifier(name='class'+str(level),outK=outK),
                      spatial_train=True,
                      intermediate_loss=False,
                      intermediate_out=4,
                      #cls_intermediate_out=2,
                      #classifier_train=True,
                      finalBlock=layers.Activation('sigmoid'),
                      forward_type='simple',
                      num_labels = labelset[0].shape[3],
#                      num_classes = 1                      
                      )

x = model.apply_full(trainset[0][0:1,:,:,:],jitter=0.05,  generate_type='random', repetitions=1000,verbose=True,scale_to_original=False)

plt.imshow(x[:,:,2])
#model.summary()
# model.save('xxx')
# model = patchwork.PatchWorkModel.load('xxx')
# model.apply_full(trainset[0][0:1,:,:,:],jitter=0.05,   repetitions=1)

#cgen.testtree(labelset[0][0:1,:,:,5:6])


#model = patchwork.PatchWorkModel.load('models/test')
#%%
model.apply_full(trainset[0][0:1,:,:,:],jitter=0.05,   repetitions=1)



#%%
#l = lambda x,y: tf.keras.losses.categorical_crossentropy(x,y,from_logits=False)
#l = tf.keras.losses.mean_squared_error
l = lambda x,y: tf.keras.losses.binary_crossentropy(x,y,from_logits=False)
loss = l


adam = tf.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=True)

model.compile(loss=loss, optimizer=adam)
cgen = model.cropper


augment = patchwork.Augmenter(    morph_width = 150
                , morph_strength=0.25
                , rotation_dphi=0.1
                , repetitions=2
                , include_original=True
                )


#%%
#model.modelname = "models/test"

model.train(trainset,labelset,
            valid_ids = [],
            augment=None,
            num_patches=11,
            epochs=100)








#%%


u, s, vh = np.linalg.svd(np.random.normal(0,1,[3,3]), full_matrices=True)
R = np.dot(u[:, :6] , vh)
print(np.matmul(R,np.transpose(R)))

















#%%

for i in range(nb):
  print("======================================================================================")
  print("iteration:" + str(i))
  print("======================================================================================")

  start = timer()
  c = cgen.sample(trainset[1:],labelset[1:],generate_type='random', num_patches=n_patches)
  end = timer()
  print("time elapsed, sampling: " + str(end - start) )

  start = timer()
  model.fit(c.getInputData(),c.getTargetData(),epochs=epochs,verbose=2)
  end = timer()
  print("time elapsed, fitting: " + str(end - start) )

  #tset = testset
  #lset = labeltestset
  tset = trainset
  lset = labelset

  numsamples = 1
  sn = 0
  numlabels = 16 # labelset.shape[3]
  sx = 9
  sy = 8
  if 1:
    if model.intermediate_loss:      
      levels = list(range(cgen.depth))
    else:
      levels = [-1]
    for n in range(sn,numsamples+sn):
      f = plt.figure(figsize=(30,30))
      cnt = 1

      start = timer()
      qq = apply(tset[n][0:1,:,:,:],levels=levels)
      
      end = timer()
      print("time elapsed, applying: " + str(end - start) )

      for l in levels:
        f.suptitle('level ' + str(l), fontsize=16)

        for k in range(numlabels):
          ax = plt.subplot(sx,sy,cnt)  
          

          q = tf.math.abs(qq[l][:,:,k]) #*math.sqrt(np.sum(l))

          maxq =  np.amax(q)
          q = q/maxq

          lab = tf.expand_dims(tf.squeeze(lset[n][0:1,:,:,k]),2)
          lab = resizeND(lab,q.shape)
          lab = tf.cast(lab,tf.float32)
          cR = tf.expand_dims(q,2)
          cG = tf.expand_dims(q,2)
          cB = tf.expand_dims(lab,2)
          a = tf.concat([cR,cG,cB],2);
          a = np.clip(a,a_min=0,a_max=1)


          ii = ax.imshow(a)
          ii.set_clim(0,25)
          ax.set_title("label " + str(k) + " max: " + str(maxq))
          ax.axis('off')
          cnt = cnt + 1
      plt.pause(0.001)






















