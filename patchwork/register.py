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
 



#%%
ddef= deformLayer(nD=nD,strength=50)
G = nifti_grid(src[0].shape,resolutions[0]['input_edges'],nD=nD)
warpy = warpLayer(src[0].shape[1:],initializer=tf.keras.initializers.Constant(src[0][0,...]),edges=resolutions[0]['input_edges'],typ='xyz',nD=nD)
wG = ddef(G)
wD = warpy(wG)[...,0:1]
target[0] = wD
resolutions[0]['output_edges'] = resolutions[0]['input_edges']
sha(target[0][0,:,:,0])

#%%

ddef = lieLayer(resolutions[0]['input_edges'],src[0].shape[1:-1],nD=2,trainable=False)
G = nifti_grid(src[0].shape,resolutions[0]['input_edges'],nD=nD)
warpy = warpLayer(src[0].shape[1:],initializer=tf.keras.initializers.Constant(src[0][0,...]),edges=resolutions[0]['input_edges'],typ='xyz',nD=nD)
wG = ddef(G)
wD = warpy(wG)[...,0:1]
target[0] = wD
resolutions[0]['output_edges'] = resolutions[0]['input_edges']
sha(target[0][0,:,:,0])


#%%

class RModel(tf.keras.Model):
    
    def __init__(self, cgen, image1=None, edges1 = None, image2=None, edges2 = None, loss=None, optimizer=None, sampling=None, lieField=None,nD=3):
        super(RModel, self).__init__()
        self.nD=nD
        self.cgen = cgen
        self.sampling = sampling        
        self.lieField = lieField

        self.grid1 = nifti_grid(image1.shape,edges1,nD=nD)
        self.grid2 = nifti_grid(image2.shape,edges2,nD=nD)
        self.image1 = image1
        self.image2 = image2
        self.edges1 = edges1
        self.edges2 = edges2 

        
        self.affine1 = affineLayer(nD=nD)
        #self.deform1 = deformLayer(nD=nD)
        
        self.deform1 = lieLayer(edges2,image2.shape[1:-1],nD=nD,**self.lieField )

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

    def sample(self,**kwargs):                        
        
        cat2 = tf.concat([self.grid2,self.image2],self.nD+1)
        
        d = self.cgen.sample(cat2,self.image1,resolutions={"input_edges":self.edges2,"output_edges":self.edges1  }, 
                                snapper=[1]*self.cgen.depth, verbose=False,**self.sampling)

        a = d.getInputData()
        
        data = []
        for k in range(self.cgen.depth):
            data.append(a['input'+str(k)])
        data = tf.concat(data,0)        
        self.patches = tf.data.Dataset.from_tensor_slices(data)
        
        self.a = a
        self.b = d.getTargetData()
        
        
    def call(self, x):
        c = x[...,0:self.nD]
        c = self.affine1(c)
        c = self.deform1(c)
        x = self.warp1(c)
        return x

    @tf.function
    def train_step(self,z):
        with tf.GradientTape() as tape:
            warped = self(z, training=True)
            x = z[...,-1]                
            loss = self.loss(x, warped[...,0])
    
        gradients = tape.gradient(loss, model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        self.train_loss(loss)
        
    
    def train_epoch(self,batch_size=64):
        
        data = self.patches.shuffle(64).batch(batch_size,drop_remainder=False)

        diter = iter(data)
        c = 1
        while True:
            z = next(diter,None)
            if z is None:
                break
            self.train_step(z)
            c = c + 1
            #print(">"+str(loss))
            print(".",end='')



# Instantiate the model
gsc_corr = 1#3

model = RModel(patchwork.CropGenerator(
                  scheme = {"destvox_rel": [1]*nD,
                            "fov_rel":[0.9]*nD,
                            "patch_size":[64]*nD
                           }, ndim=nD, depth=2), 
              sampling = {"generate_type":'random',
                         "branch_factor":32,
                         "num_patches":4 },
              lieField = {"sfac":10,"sens":0.01},
               
              loss=tf.keras.losses.MeanSquaredError(),
              optimizer= tf.keras.optimizers.Adam(learning_rate=0.01),
              
              
              image1=target[0],edges1=resolutions[0]['output_edges'],
              image2=src[0]/gsc_corr,   edges2=resolutions[0]['input_edges'], nD=nD)

    
#xx =model(model.patches)

#k = 0
#sh(model.a['input0'][k,:,:,-1])
#plt.pause(0.001)
#sh(model.b[0][k,:,:,0])
#plt.pause(0.001)
#sh(xx[k,:,:,0])

#%%
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    0.005,
    decay_steps=5,
    decay_rate=0.99,
    staircase=True) 

model.optimizer= tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# Define the training loop
def fit(epochs):
    
    for epoch in range(epochs):        
        model.train_loss.reset_states()
        if epoch%4==0:
            model.sample()
            
            r =0#round(np.random.uniform(low=0,high=model.patches.shape[0]-1))
            #warped = model(model.patches,training=True)
            #sh(warped[r,:,:,0])
            #plt.pause(0.2)
            #sh(model.patches[r,:,:,-1])

            wim1 = model(model.grid2,training=False)
            sha(wim1[...,0:1])
            sha(wim1[...,0:1] - src[0])
            plt.pause(0.0001)


        model.train_epoch()

        # Display metrics at the end of each epoch
        print(f'Epoch {epoch + 1}, '
              f'Loss: {model.train_loss.result()}, ')
     #   print("learning rate: " + str(model.optimizer._decayed_lr(tf.float32).numpy()))

# Call the train function with your training dataset and the number of epochs
fit(1000)

#%%

xx =model(a['input0'])

k = 0
sh(a['input0'][k,:,:,16,3])
plt.pause(0.001)
sh(xx[k,:,:,16,0])


