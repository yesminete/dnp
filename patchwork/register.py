#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 09:37:52 2023

@author: kellnere
"""



import numpy as npss
import tensorflow as tf
import math
from tensorflow.keras import Model
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import patchwork 
from patchwork import improc_utils
from patchwork.customLayers import *
from patchwork.improc_utils import *
import patchwork.model as patchwork

sh = lambda x,**kwargs: (plt.imshow(tf.squeeze(x),**kwargs,vmax=2000), plt.pause(0.00001))
sha = lambda x,**kwargs: (plt.imshow(tf.squeeze(x),**kwargs), plt.pause(0.00001))

nD = 2


if nD == 3:
    contrasts = [ { 'subj1' :  '/nfs/noraimg/data1/SANDBOX/undefined_undefined_100307/NII_20230315/v1.nii' } ]
    labels   = [ { 'subj1' :  '/nfs/noraimg/data0/HCP2/analysis/ANALYSIS_degrade/stack/mean_T1w_restore.nii' } ]
    src,target,resolutions,subjs  = load_data_structured(contrasts=contrasts,labels=labels,subjects=['subj1']
                                                     ,max_num_data=1,reslice_labels=False)
    cgen = patchwork.CropGenerator(
                  scheme = {
                      #"destvox_mm": [1,1,1],
                      "destvox_rel": [1]*nD,
                      #"fov_mm":[10,10,10],
                      "fov_rel":[0.9]*nD,
                      "patch_size":[32]*nD
                      },
                  ndim=nD,
                  interp_type = 'NN',
                  scatter_type = 'NN',
                  #system='matrix',
                  depth=4)


if nD == 2:
    
    contrasts = [ { 'subj1' :  '/software/patchwork_master/tests/example2d.nii.gz' } ]
    labels   = [ { 'subj1' :  '/software/patchwork_master/tests/example2d_b.nii.gz'  } ]
    src,target,resolutions,subjs  = load_data_structured(contrasts=contrasts,labels=labels,subjects=['subj1']
                                                         ,max_num_data=1,nD=2,reslice_labels=False)

    src[0] = src[0][:,30:-30,:]
    target[0] = target[0][:,30:-30,:]
    cgen = patchwork.CropGenerator(
                  scheme = {
                      #"destvox_mm": [1,1,1],
                      "destvox_rel": [1]*nD,
                      #"fov_mm":[10,10,10],
                      "fov_rel":[1]*nD,
                      "patch_size":[64]*nD
                      },
                  ndim=nD,
                  interp_type = 'NN',
                  scatter_type = 'NN',
                  #system='matrix',
                  depth=2)



#%%
class deformLayer(layers.Layer):

  def __init__(self,fdims=[32,32,32],nD=3,scale=10,strength=5,**kwargs):
    super().__init__(**kwargs)
    self.scale = scale
    self.strength = strength
    self.nD = nD
    self.fdims = fdims
    self.conv = []
    if nD == 2:
        cop = lambda n,**kwargs: layers.Conv2D(n,1,**kwargs)
    else:
        cop = lambda n,**kwargs: layers.Conv3D(n,1,**kwargs)
    for k in self.fdims:
        self.conv.append(cop(k,use_bias=True))
        #                     kernel_initializer=tf.keras.initializers.RandomNormal,
         #                    bias_initializer=tf.keras.initializers.RandomNormal))
    self.conv.append(layers.BatchNormalization())
    self.conv_final=cop(nD,use_bias=False)

  def call(self, image):
      
     x = image/self.scale
     for k in range(len(self.conv)):
         x = self.conv[k](x)
         x = tf.math.cos(x)
         
     x = self.conv_final(x)    
     x = image+self.strength*x
     return x

ddef= deformLayer(nD=nD,strength=40)
G = nifti_grid(src[0].shape,resolutions[0]['input_edges'],nD=nD)
warpy = warpLayer(src[0].shape[1:],initializer=tf.keras.initializers.Constant(src[0][0,...]),edges=resolutions[0]['input_edges'],typ='xyz',nD=nD)
wG = ddef(G)
wD = warpy(wG)[...,0:1]
target[0] = wD
resolutions[0]['output_edges'] = resolutions[0]['input_edges']
sha(target[0][0,:,:,0])
#%%    
class affineLayer(layers.Layer):

  def __init__(self,nD=3,**kwargs):
    super().__init__(**kwargs)
    def initg(shape,dtype=None):
        return tf.random.normal(shape)*0
    def initt(shape,dtype=None):
        return tf.random.normal(shape)*0

    self.nD = nD
    self.gen = self.add_weight(shape=[nD,nD], 
                        initializer=initg, trainable=True,name=self.name)    
    self.trans= self.add_weight(shape=[nD], 
                        initializer=initt, trainable=True,name=self.name)    

  def call(self, image):
     A = tf.linalg.expm(self.gen)
     x = tf.einsum('mn,...n->...m',A,image) + self.trans
     return x
         


class RModel(tf.keras.Model):
    
    def __init__(self, cgen, image1=None, edges1 = None, image2=None, edges2 = None, loss=None, optimizer=None,nD=3):
        super(RModel, self).__init__()
        self.nD=nD
        self.cgen = cgen
        

        self.grid1 = nifti_grid(image1.shape,edges1,nD=nD)
        self.grid2 = nifti_grid(image2.shape,edges2,nD=nD)
        self.image1 = image1
        self.image2 = image2
        self.edges1 = edges1
        self.edges2 = edges2 

        
        self.affine1 = affineLayer(nD=nD)
        self.deform1 = deformLayer(nD=nD)
        self.warp1 = warpLayer(image1.shape[1:],initializer=tf.keras.initializers.Constant(image1[0,...]),edges=edges1,typ='xyz',nD=nD)
        
        if loss is None:
            self.loss=tf.keras.losses.MeanSquaredError()
        else:
            self.loss=loss
                
        if optimizer is None:
            self.optimizer= tf.keras.optimizers.Adam(learning_rate=0.001)
        else:
            self.optimizer=optimizer
            
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')

    def sample(self):                        
        
        cat2 = tf.concat([self.grid2,self.image2],self.nD+1)
        
        d = cgen.sample(cat2,self.image1,generate_type='random',branch_factor=2,
                                resolutions={"input_edges":self.edges2,"output_edges":self.edges1  }, # resolutions[0],
                                snapper=[1]*cgen.depth, num_patches=2, verbose=False)

        a = d.getInputData()
        
        data = []
        for k in range(self.cgen.depth):
            data.append(a['input'+str(k)])
        data = tf.concat(data,0)        
        self.patches = data
        
        self.a = a
        self.b = d.getTargetData()
        
        
    def call(self, x):
        c = x[...,0:self.nD]
        c = self.affine1(c)
        c = self.deform1(c)
        x = self.warp1(c)
        return x
    

    def train_step(self):
        
        data = self.patches
        with tf.GradientTape() as tape:
            warped = self(data, training=True)
            x = data[...,-1]
            
            loss = self.loss(x, warped[...,0])
        
        gradients = tape.gradient(loss, model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        self.train_loss(loss)



# Instantiate the model
gsc_corr = 1#3

model = RModel(cgen, nD=nD,
               loss=tf.keras.losses.MeanSquaredError(),
               optimizer= tf.keras.optimizers.Adam(learning_rate=0.001),
               image1=target[0],edges1=resolutions[0]['output_edges'],
               image2=src[0]/gsc_corr,   edges2=resolutions[0]['input_edges'])

model.sample()
    
xx =model(model.patches)

k = 0
sh(model.a['input0'][k,:,:,-1])
plt.pause(0.001)
sh(model.b[0][k,:,:,0])
plt.pause(0.001)
sh(xx[k,:,:,0])

#%%
              
# Define the training loop
def train(epochs):
    
    #step = tf.function(model.train_step)
    step = model.train_step
    for epoch in range(epochs):        
        model.train_loss.reset_states()

        step()

        if epoch%10==0:
            r =0#round(np.random.uniform(low=0,high=model.patches.shape[0]-1))
            #warped = model(model.patches,training=True)
            #sh(warped[r,:,:,0])
            #plt.pause(0.2)
            #sh(model.patches[r,:,:,-1])

            wim1 = model(model.grid2,training=False)
            sha(wim1[...,0:1] - src[0])
            plt.pause(0.0001)

        # Display metrics at the end of each epoch
        print(f'Epoch {epoch + 1}, '
              f'Loss: {model.train_loss.result()}, ')

# Call the train function with your training dataset and the number of epochs
train(200)

#%%

xx =model(a['input0'])

k = 0
sh(a['input0'][k,:,:,16,3])
plt.pause(0.001)
sh(xx[k,:,:,16,0])


