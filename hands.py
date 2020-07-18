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

import patchwork.model as patchwork
from patchwork.improc_utils import *

from DPX_core import *
from DPX_improc import *
import customLayers


#%%

trainset,labelset,resolutions,subjs = load_data_for_training(
                                project = 'tschaga',    
                                subjects = '#Tag:test',
                               # subjects = '15341572', #'#Tag:MIDItrain',
                                contrasts_selector = ['T1.nii.gz'],
                                labels_selector = ['mask_untitled0.nii.gz','mask_untitled0.nii.gz'],
                                one_hot_index_list=[0,1],
#                                labels_selector = ['anno*/testset.ano.json'],
                              #  labels_selector = ['forms/untitledrrr.form.json'],
                              #  annotations_selector = { 'labels' : [ [ 'XYZ.A1', 'XYZ.A2' ] , ['WW.noname'] ] 
                              #                         ,'sizefac':1,
                              #                           'classes' : [ 'mycheck' , 'myrate.AF']                                                      
                              #                         },
                                exclude_incomplete_labels=True,
                                add_inverted_label=False,
                                max_num_data=1
                                )




#%%




trainset = []
labelset = []
resolutions = []
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

  labelset.append( tf.convert_to_tensor(a[...,8:9],dtype=tf.float32)   )
  
  resolutions.append(img.header['pixdim'][1:4])
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
u = customLayers.createUnet3D_v1(feature_dim=[10,20,50],depth=3,verbose=True); r = u(tf.ones([1,32,32,32,1])); r.shape

#%%

################################################################################################




biConvolution = patchwork.biConvolution

def BNrelu(name=None):
  return [layers.BatchNormalization(name=name), layers.LeakyReLU()]

n = 2
nD = 2
s = (2,2)
pooly = lambda: layers.MaxPooling3D(pool_size=s)

def conv_down(name=None,dest=None):
  x = biConvolution(n,3,name=name,nD=nD)
  x.dest = dest
  #return layers.Conv2D(n,3,padding='SAME') 
  return x
def conv_up(name=None):
  #return layers.Conv2DTranspose(n,3,padding='SAME',strides=(2,2)) 
  return biConvolution(n,3,transpose=True,strides=s,name=name,nD=nD)
def conv_out(outK,name=None):
  #return layers.Conv2D(outK,3,padding='SAME') 
  return biConvolution(outK,3,name=name,nD=nD)

def createBlock_(name=None,depth=2,outK=1):
  block = patchwork.CNNblock(name=name)
  for z in range(depth):
    id_d = name+str(1000 + z+1)
    id_u = name+str(2000 + depth-z+1)
    block.add( conv_down(id_d+"1conv")) #  conv_down(name=id_d+"2conv",  dest=id_u+"relu" ) )
    block.add( BNrelu(name=id_d+"2relu") + [pooly() ] )    
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
    theLayers[id_d+"relu"] = BNrelu() + [pooly() ]
    theLayers[id_u+"conv"] =  [conv_up()]
    theLayers[id_u+"relu"] = BNrelu()
  theLayers["3000"] =  [layers.Dropout(rate=0.5), conv_out(outK)]
  return patchwork.CNNblock(theLayers)

def createClassifier(name=None,depth=4,outK=2):
  theLayers = {}
  for z in range(depth):
    id_d = str(1000 + z+1)
    theLayers[id_d+"conv"] = conv_down()
    theLayers[id_d+"relu"] = BNrelu() + [pooly() ]
  theLayers["3001"] =  layers.Flatten()
  theLayers["3002"] =  layers.Dense(outK)
  theLayers["3003"] = layers.Activation('sigmoid')
    
  return patchwork.CNNblock(theLayers)


#x = createBlock_(name="aa",outK=3)
#print(x(trainset[0][0:1,0:32,0:32,:]).shappatchwork.CNNblock()e)
#y = createBlock(outK=3)
#print(y(trainset[0][0:1,0:32,0:32,:]).shape)
#%% 3D
nD=3
cgen = patchwork.CropGenerator(patch_size = (32,32,32), 
                  scale_fac =  0.5, 
                  scale_fac_ref = 'max',
                  init_scale = -1,#'50mm,50mm,50mm',
                  #smoothfac_data=['boxcar',None],
                  ndim=3,
                  interp_type = 'NN',
                  scatter_type = 'NN',
                  
                  #create_indicator_classlabels=True,
                  depth=2)



# cgen.sample(trainset[0],None,generate_type='random',
#                                     resolutions=[1,1,1],
#                                     num_patches=1,
#                                     verbose=True)


model = patchwork.PatchWorkModel(cgen,
                      #blockCreator= lambda level,outK : createBlock_(name='block'+str(level),outK=outK),
                      blockCreator= lambda level,outK : customLayers.createUnet_v1(3,outK=outK,nD=nD),
                     # preprocCreator = lambda level: patchwork.normalizedConvolution(nD=2),
                      spatial_train=True,
                      intermediate_loss=False,
                      #block_out=[4,1],

                    #  classifierCreator = lambda level,outK: createClassifier(name='class'+str(level),outK=outK),
                    #  cls_intermediate_out=2,
                    #  cls_intermediate_loss=True,
                    #  classifier_train=True,
                      
                      finalBlock= customLayers.sigmoid_softmax(),
                     # forward_typinverse_rote='simple',
                      num_labels = 1,
#                      num_classes = 1                      
                      )
# #%
# x = model.apply_full(trainset[0][0:1,...],resolution=resolutions[0],
#                      jitter=1,jitter_border_fix=False, generate_type='tree', repetitions=1,verbose=True,scale_to_original=False,
#                      lazyEval=0.3#{'reduceFun':'classifier_output'}
#                      )

res = model.apply_full(trainset[0],resolution=resolutions[0],
                       generate_type='random',jitter=0,   repetitions=500,dphi=0.05,verbose=True,scale_to_original=False)
#res = model.apply_full(trainset[0][0:1,0:300,0:300,...],resolution=resolutions[0],

plt.imshow(tf.squeeze(res[30,:,:]))

#%% 2D
nD=2
cgen = patchwork.CropGenerator(patch_size = (128,128) ,
                  scale_fac =  0.8,
                  scale_fac_ref = 'min',
                  init_scale = '140mm,140mm',
                  #smoothfac_data=['boxcar',0.5],
                  ndim=nD,
                  interp_type = 'NN',
                  keepAspect=True,
                  scatter_type = 'NN',
                  #create_indicator_classlabels=True,
                  depth=1)



# cgen.sample(tf.ones([1,750,750]),None,generate_type='tree',
#                                     resolutions=[1,1],
#                                     num_patches=1,
#                                     verbose=True)





model = patchwork.PatchWorkModel(cgen,
                      blockCreator= lambda level,outK : customLayers.createUnet_bi(3,outK=outK,nD=nD),
                     # preprocCreator = lambda level: patchwork.normalizedConvolution(nD=2),
                      spatial_train=True,
                      intermediate_loss=False,
                      #block_out=[4,1],
                          
                    #  classifierCreator= lambda level,outK : customLayers.simpleClassifier(outK=outK,nD=nD),
                    #  classifierCreator = lambda level,outK: createClassifier(name='class'+str(level),outK=outK),
                    #  cls_intermediate_out=2,
                    #  cls_intermediate_loss=True,
                     # classifier_train=True,
                      
                    #  finalBlock= customLayers.sigmoid_softmax(),
                     # forward_typinverse_rote='simple',
                      #num_labels = 4,
                       num_classes = 4                      
                      )
# #%
# x = model.apply_full(trainset[0][0:1,...],resolution=resolutions[0],
#                      jitter=1,jitter_border_fix=False, generate_type='tree', repetitions=1,verbose=True,scale_to_original=False,
#                      lazyEval=0.3#{'reduceFun':'classifier_output'}
#                      )

res = model.apply_full(trainset[0][0:1,:,:,...],resolution=resolutions[0],
                       generate_type='tree',jitter=0,   repetitions=1,dphi=0.1,verbose=True,scale_to_original=True,testIT=False)
#res = model.apply_full(trainset[0][0:1,0:300,0:300,...],resolution=resolutions[0],
#                       generate_type='random',jitter=0.05,   repetitions=20,dphi=0.9,verbose=True,scale_to_original=False)

print(res.shape)
plt.imshow(tf.squeeze(res))
plt.pause(0.001)
#print(tf.reduce_sum(tf.math.abs(res-tf.squeeze(trainset[0]))).numpy()/100000)inverse_rot
#plt.imshow(x[...,0],vmin=0,vmax=0.0000000001)

#model.summary()
# model.save('xxx')
# model = patchwork.PatchWorkModel.load('xxx')
# model.apply_full(trainset[0][0:1,:,:,:],jitter=0.05,   repetitions=1)

#cgen.testtree(labelset[0][0:1,:,:,5:6])

#model = patchwork.PatchWorkModel.load('models/test')
#%%
res = model.apply_full(trainset[0][0:1,...],jitter=0.05,   repetitions=1,verbose=True)


#%%
#l = lambda x,y: tf.keras.losses.categorical_crossentropy(x,y,from_logits=False)
#l = tf.keras.losses.mean_squared_error

def cc(x,y):
    sx = tf.reduce_sum(x,axis=-1,keepdims=True)
    sx = tf.concat([x,1-sx],len(sx.shape)-1)
    sy = tf.reduce_sum(y,axis=-1,keepdims=True)
    sy = tf.concat([y,1-sy],len(sx.shape)-1)
    return tf.keras.losses.categorical_crossentropy(x,y,from_logits=False)
    
    
l = lambda x,y: tf.keras.losses.binary_crossentropy(x,y,from_logits=False)
l_logits = lambda x,y: tf.keras.losses.binary_crossentropy(x,y,from_logits=True)
loss = [l_logits,l]
loss = l

augment = patchwork.Augmenter(    morph_width = 150
                , morph_strength=0.25
                , rotation_dphi=0.1
                , repetitions=1
                , include_original=True
                )


#%%
#model.modelname = "models/test"

model.train(trainset,labelset,
            resolutions=resolutions,
            loss=loss,
            valid_ids = [],
            augment=None,
            #balance={'ratio':0.3,'label_range':range(2),'label_weight':[1,0.2]},
            num_patches=10,
            epochs=5)








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






















