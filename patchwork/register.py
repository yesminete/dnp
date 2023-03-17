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
import patchwork2 
from patchwork2 import improc_utils
from patchwork2.customLayers import *
from patchwork2.improc_utils import *
import patchwork2.model as patchwork

sh = lambda x,**kwargs: (plt.imshow(tf.squeeze(x),**kwargs,vmax=2000), plt.pause(0.00001))
sha = lambda x,**kwargs: (plt.imshow(tf.squeeze(x),**kwargs), plt.pause(0.00001))


nD=3
contrasts = [ { 'subj1' :  '/nfs/noraimg/data1/SANDBOX/undefined_undefined_100307/NII_20230315/v1.nii' } ]
labels   = [ { 'subj1' :  '/nfs/noraimg/data0/HCP2/analysis/ANALYSIS_degrade/stack/mean_T1w_restore.nii' } ]
src,target,resolutions,subjs  = load_data_structured(contrasts=contrasts,labels=labels,subjects=['subj1']
                                                     ,max_num_data=1,reslice_labels=False)

             


#%
#% 3D
nD=3
cgen = patchwork.CropGenerator(
                  scheme = {
                      #"destvox_mm": [1,1,1],
                      "destvox_rel": [1,1,1],
                      #"fov_mm":[10,10,10],
                      "fov_rel":[0.9,0.9,0.9],
                      "patch_size":[32,32,32]
                      },
                  ndim=nD,
                  interp_type = 'NN',
                  scatter_type = 'NN',
                  #system='matrix',
                  depth=4)






#%%
class deformLayer(layers.Layer):

  def __init__(self,fdims=[32,32,32],nD=3,**kwargs):
    super().__init__(**kwargs)

    self.nD = nD
    self.fdims = fdims
    self.conv = []
    for k in self.fdims:
        self.conv.append(layers.Conv3D(k,1,use_bias=True))
        #self.conv.append(layers.BatchNormalization())
    self.conv_final=layers.Conv3D(3,1)

  def call(self, image):
      
     x = image
     for k in range(len(self.conv)):
         x = self.conv[k](x)
         x = tf.math.cos(x)
         
     x = image+5*self.conv_final(x)    
     return x
     
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
        

        self.grid1 = nifti_grid(image1.shape,edges1)
        self.grid2 = nifti_grid(image2.shape,edges2)
        self.image1 = image1
        self.image2 = image2
        self.edges1 = edges1
        self.edges2 = edges2 

        
        self.affine1 = affineLayer()
        self.deform1 = deformLayer()
        self.warp1 = warpLayer(image1.shape[1:],initializer=tf.keras.initializers.Constant(image1[0,...]),edges=edges1,typ='xyz')
        
        if loss is None:
            self.loss=tf.keras.losses.MeanSquaredError()
        else:
            self.loss=loss
                
        if optimizer is None:
            self.optimizer= tf.keras.optimizers.Adam(learning_rate=0.01)
        else:
            self.optimizer=optimizer
            
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')

    def sample(self):                        
        
        cat2 = tf.concat([self.grid2,self.image2],self.nD+1)
        
        d = cgen.sample(cat2,self.image1,generate_type='random',branch_factor=1,
                                resolutions={"input_edges":self.edges2,"output_edges":self.edges1  }, # resolutions[0],
                                snapper=[1]*cgen.depth, num_patches=32, verbose=False)

        a = d.getInputData()
        
        data = []
        for k in range(self.cgen.depth):
            data.append(a['input'+str(k)])
        data = tf.concat(data,0)        
        self.patches = data
        
        self.a = a
        self.b = d.getTargetData()
        
        
    def call(self, x):
        c = x[...,0:3]
        #c = self.affine1(c)
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
model = RModel(cgen, 
               loss=tf.keras.losses.MeanSquaredError(),
               optimizer= tf.keras.optimizers.Adam(learning_rate=0.01),
               image1=target[0],edges1=resolutions[0]['output_edges'],
               image2=src[0]/3,   edges2=resolutions[0]['input_edges'])

model.sample()
    
xx =model(model.patches)

k = 0
sh(model.a['input0'][k,:,:,16,3])
plt.pause(0.001)
sh(model.b[0][k,:,:,16,0])
plt.pause(0.001)
sh(xx[k,:,:,16,0])

#%%
              
# Define the training loop
def train(epochs):
    
    #step = tf.function(model.train_step)
    step = model.train_step
    for epoch in range(epochs):        
        model.train_loss.reset_states()

        step()

        if epoch%10==0:
            r =round(np.random.uniform(low=0,high=model.patches.shape[0]-1))
            warped = model(model.patches,training=True)
            sha(warped[r,:,:,16,1])
            sh(warped[r,:,:,16,0])
            plt.pause(0.2)
            sh(model.patches[r,:,:,16,-1])
    
           # print(model.conv1.weights)
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


