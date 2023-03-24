#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 09:37:52 2023

@author: kellnere
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = "3"

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





def get_gpu_memory():
     try:    
         import subprocess as sp    
         import os
         _output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
         COMMAND = "nvidia-smi --query-gpu=memory.free --format=csv"
         memory_free_info = _output_to_list(sp.check_output(COMMAND.split()))[1:]
         memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
         dev = int(os.environ["CUDA_VISIBLE_DEVICES"])
         if dev < len(memory_free_values):
             return memory_free_values[dev]
         else:
             return None
       
     except:
         return None

print(get_gpu_memory())

#%%

sh = lambda x,**kwargs: (plt.imshow(tf.squeeze(x),**kwargs,vmax=2000), plt.pause(0.00001))
sha = lambda x,**kwargs: (plt.imshow(tf.squeeze(x),**kwargs), plt.pause(0.00001))

nD = 3

if nD == 3:
    contrasts = [ { 'subj1' :  '/nfs/noraimg/data1/SANDBOX/undefined_undefined_100307/NII_20230315/v1.nii' } ]
    labels   = [ { 'subj1' :  '/nfs/noraimg/data0/HCP2/analysis/ANALYSIS_degrade/stack/mean_T1w_restore.nii' } ]
    src,target,resolutions,subjs  = load_data_structured(contrasts=contrasts,labels=labels,subjects=['subj1']
                                                     ,max_num_data=1,reslice_labels=False)
  

if nD == 2:
    
    contrasts = [ { 'subj1' :  '/software/patchwork_master/tests/example2d.nii.gz' } ]
    labels   = [ { 'subj1' :  '/software/patchwork_master/tests/example2d_b.nii.gz'  } ]
    src,target,resolutions,subjs  = load_data_structured(contrasts=contrasts,labels=labels,subjects=['subj1']
                                                         ,max_num_data=1,nD=2,reslice_labels=False)

    src[0] = src[0][:,30:-30,:]
    target[0] = target[0][:,30:-30,:]
 


#%%

ddef = lieLayer(resolutions[0]['input_edges'],src[0].shape[1:-1],nD=nD,trainable=False,
                 sens=0.003, sfac=[2,4,8,16,32,64])

G = nifti_grid(src[0].shape,resolutions[0]['input_edges'],nD=nD)
warpy = warpLayer(src[0].shape[1:],initializer=tf.keras.initializers.Constant(src[0][0,...]),edges=resolutions[0]['input_edges'],typ='xyz',nD=nD)
wG = ddef(G)
wD = warpy(wG)[...,0:1]

target[0] = wD
resolutions[0]['output_edges'] = resolutions[0]['input_edges']
sha(target[0][0,:,:,0])

#%%
ddef= deformLayer(nD=nD,strength=100)
G = nifti_grid(src[0].shape,resolutions[0]['input_edges'],nD=nD)
warpy = warpLayer(src[0].shape[1:],initializer=tf.keras.initializers.Constant(src[0][0,...]),edges=resolutions[0]['input_edges'],typ='xyz',nD=nD)
wG = ddef(G)
wD = warpy(wG)[...,0:1]
target[0] = wD
resolutions[0]['output_edges'] = resolutions[0]['input_edges']
sha(target[0][0,:,:,0])

#%%

class RModel(tf.keras.Model):
    
    def __init__(self, cgen, image1=None, edges1 = None, image2=None, edges2 = None, 
                 loss=None, optimizer=None, sampling=None, lieField=None,affine=None,
                 nD=3):
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

        
        self.affine1 = affineLayer(nD=nD,**affine)
       
        
        self.deform1 = lieLayer(edges2,image2.shape[1:-1],nD=nD,**self.lieField)

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
        
        d = self.cgen.sample(cat2,None,resolutions={"input_edges":self.edges2,"output_edges":self.edges1  }, 
                                snapper=[1.05]+[1]*(self.cgen.depth-1), verbose=False,**self.sampling)

        a = d.getInputData()
        
        data = []
        for k in range(self.cgen.depth):
            data.append(a['input'+str(k)])
            
        datacat = tf.concat(data,0) 
        self.patches = tf.data.Dataset.from_tensor_slices(datacat)
        self.tmp = datacat
        self.crop = d
        self.data = data
        
#        self.a = a
#        self.b = d.getTargetData()
        
        
    def thewarp(self,x):
        c = x[...,0:self.nD]
        c = self.affine1(c)
        c = self.deform1(c)
        return c
        
    def call(self, x):
        c = self.thewarp(x)
        x = self.warp1(c)
        return x

    def train_step(self,z):
        with tf.GradientTape() as tape:
            warped = self(z, training=True)
            mask = warped[...,self.nD]
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
          #  self.deform1.distort(0.0)
            c = c + 1
            #print(">"+str(loss))
            print(".",end='')



# Instantiate the model
gsc_corr = 3
edges_src = resolutions[0]['input_edges']
edges_target = resolutions[0]['output_edges']
#edges = np.matmul(np.diag([-1,-1,1,1]),edges)

model = RModel(patchwork.CropGenerator(
                  scheme = {"destvox_rel": [1]*nD,
                            "fov_rel":[0.7]*nD,
                            "patch_size":[32]*nD
                           }, ndim=nD, depth=3), 
#              sampling = {"generate_type":'random',
#                         "branch_factor":6,
#                         "num_patches":4 },
              sampling = {"generate_type":'random',
                         "branch_factor":4,
                         "num_patches":8 },
              
              #lieField = {"sfac":[2,4,8,16,32,64],"sens":0.0005},  # 2D
              lieField = {"sfac":[2,4,8,16],"sens":0.0001},
              affine = {"sensA":1,"sensT":100},
              #loss=tf.keras.losses.MeanSquaredError(),
              loss=lambda x,y: MutualInfoGaussian(x,y,nD=nD),
              
              
              image1=target[0],edges1=edges_target,
              image2=src[0]/gsc_corr,   edges2=edges_src, nD=nD)

    

#xx =model(model.patches)

#k = 0
#sh(model.a['input0'][k,:,:,-1])
#plt.pause(0.001)
#sh(model.b[0][k,:,:,0])
#plt.pause(0.001)
#sh(xx[k,:,:,0])1





model.sample()

_,cov = model.crop.stitchResult(model.data,-1)
if nD==2:
    sha(cov)

x = model(model.tmp)
#for k in range(x.shape[0]):
#    sha(x[k,:,:,0])



#%
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    0.005,
    decay_steps=1,
    decay_rate=0.9999,
    staircase=True) 

model.optimizer= tf.keras.optimizers.Adam(learning_rate=lr_schedule)#,beta_1=0.9)#beta_2=0.1)

model.sample()

#%%
# Define the training loop
def fit(epochs):
    
    for epoch in range(epochs):        
        model.train_loss.reset_states()
        if epoch%5==0:
            
           # model.sample()
           # r =0#round(np.random.uniform(low=0,high=model.patches.shape[0]-1))
            #warped = model(model.patches,training=True)
            #sh(warped[r,:,:,0])
            #plt.pause(0.2)
            #sh(model.patches[r,:,:,-1])

            wim1 = model(model.grid2,training=False)
            sh(src[0][...,90,0:1]/gsc_corr)
            sh(wim1[...,90,0:1])
            sha(wim1[...,90,0:1] - src[0][...,90,0:1]/gsc_corr,vmin=-500,vmax=500)
            plt.pause(0.0001)
            #wim1 = model.thewarp(model.grid2)
            #sha(wim1[...,0:1])
            #plt.pause(0.0001)


        model.train_epoch()

        # Display metrics at the end of each epoch
        print(f'Epoch {epoch + 1}, '
              f'Loss: {model.train_loss.result()}, ')
        print("learning rate: " + str(model.optimizer._decayed_lr(tf.float32).numpy()))

# Call the train function with your training dataset and the number of epochs
fit(250)


#%%

#wim1 = model(model.grid2,training=False)
for sl in range(30,150,5):
    sha(wim1[0,:,sl,:,0:1] - src[0][0,:,sl,:,0:1]/gsc_corr,vmin=-500,vmax=500)
    sh(wim1[0,:,sl,:,0:1])
    sh(src[0][0,:,sl,:,0:1]/gsc_corr)



#%%

xx =model(a['input0'])

k = 0
sh(a['input0'][k,:,:,16,3])
plt.pause(0.001)
sh(xx[k,:,:,16,0])


