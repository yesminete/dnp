#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 20:43:48 2021

@author: reisertm
"""

import sys 
import os
import matplotlib as mpl
import matplotlib.pyplot as plt


import numpy as npss
import tensorflow as tf
import math
from tensorflow.keras import Model
from tensorflow.keras import layers
import matplotlib.pyplot as plt



sys.path.append("/software")
import patchwork2 as patchwork



#%%

#% definition data sources


# define your data sources 
contrasts = [ { 'subj1' :  'example2d.nii.gz',  
                'subj2' :  'example2d_b.nii.gz' , 
                'subj3' :  'example2d_b.nii.gz'  
                } ]
labels   = [  { 'subj1' :  'example2d_label.nii.gz', 
                'subj2' :  'example2d_label_b.nii.gz',
                'subj3' :  'example2d_label_b.nii.gz'} ]

subjects = [ 'subj1', 'subj2', 'subj3'];

# define ou want some validation dta
valid_ids = []

modelfi = "models/yourmodel"

reinit_model = True




#% definition of your problem


# dim of problem (2D/3D)
nD = 2


### PATCHING OPTIONS

# - depth: depth of patchwork

# - scheme concludes the parameter of the way patches are drawn
# - patch_size specifies the pixel/voxel dimensions of the path by a 
#   tuple (2D) or a triple (3D), or a list of tuples/triples if the 
#   patch sizes are dependend on depth.
# - destvox_mm or destvox_rel are mutual exclusive and determine the 
#   voxel sizes of the output of the final network reponse, either in 
#   millimeter (mm) or relative to input voxel sizes.
# - fov_mm and fov_rel determine the field of view of the network, i.e. 
#   the size of patch in the coarsest level, either in millimiter (mm) or 
#   relative to the total size of the input.
# - smoothfac_data: how data is processed before patching: 
#   None : nothing 
#   float : image is smoothed by Gaussian of this width
#   "max" : a maxpooling of appropriate size is applied
#   "boxcar" : simple avg pooling
#   "mixture" :  concat of maxpool,minpool,boxcar
# - smoothfac_label:
#   same like for data but no mixture
# - interp_type: nearest Neighbor (NN) or linear (lin)
# - scatter_type: nearest Neighbor (NN) or linear (lin)
# - normalize_input: None or "max" at the moment.

patching = {        
    "depth":2,                    
    "scheme":{ 
        "patch_size":[128,128],                
        "destvox_mm": None,
        "destvox_rel":[1,1],
        "fov_mm":None,
        "fov_rel":[0.5,0.5],
     },
    "smoothfac_data" : 0,   
    "smoothfac_label" : 0, 
    "interp_type" : "NN",    
    "scatter_type" : "NN",
    "normalize_input" : None,
    }

### NETWORK OPTIONS
# - blockCreator: function returning the CNN for the individual levels. 
#   Just pass your favourite keras layer, or a patchwork.CNNblock.
#   The network may depend on the patching level.
# - intermediate_out: number of features passed from one level to the other
# - intermediate_loss: True/False, determines wheter the loss is also taken
#   at the intermedite output levels

network = {    
    "blockCreator": lambda level,outK,input_shape : 
        patchwork.customLayers.createUnet_v2(depth=5,outK=outK,nD=nD,input_shape=input_shape,feature_dim=[8,8,8,8,8],dropout=True),            
    "classifierCreator": lambda level,outK: patchwork.customLayers.simpleClassifier(outK=outK,nD=nD,depth=4),
#    "preprocCreator": lambda level: patchwork.customLayers.HistoMaker(trainable=True,init='ct',dropout=0,nD=nD,normalize=False),    
     "finalBlock":layers.Activation('sigmoid'), 
    "intermediate_out":8,
    "intermediate_loss":True,          
    }

## DATA IMPORT OPTIONS
# - align_physical: True/False, input arrays are permuted/flipped such that the affine
#   has absolute maximas on the diagonal and the diagonal entries are positive
# - crop_fdim: list, feature dimensions which are cropped out of the input
# - crop_fdim_labels: list, same for the labels
# - crop_only_nonzero: take only those label dimensions for learning that 
#   contain non-zero labels
# - threshold: float; threshold for the label images. If "None", no thresholding
#   operation is applied
# - add_inverted_label: use the inverted union of all labels as last label
# - one_hot_index_list: list, if input labels are class indices, use this to map to
#   "one hot" representation.
# - annotations_selector: if you load an annotation (*.ano.json) you specfify here how
#   annotations are translated to label images

loading = {
    "annotations_selector": None,
    "align_physical":False,
    "crop_fdim":None,
    "crop_fdim_labels":None,
    "crop_only_nonzero":False,
    "threshold":0.5,
    "add_inverted_label":False,
    "one_hot_index_list":None
    }


## TRAINING OPTIONS
# - num_patches: number of patches per training image
# - balance: to control balance of non-label vs. label patches (to explain..)
# - epochs: number of erpochs per iteration (patching instance)
# - num_its: numner of total iteration
# - samples_per_it: if memory is not enough, use this to restrict number of 
#   training images per iteration. 
# - reload_after_it: use this to determine, when new training images are loaded



training = {
   "num_patches":2,
   #"augment": {"dphi":0.2, "flip":[1,0] , "dscale":[0.1,0.1] },
   "epochs":50,
   "num_its":100,                
   "loss": tf.keras.losses.binary_crossentropy,
   #"hard_mining":0.1,
   #"hard_mining_maxage":50,
   "reload_after_it":5,
   "samples_per_it":15,
   }


        




# (and some legacy)

reload_after_it = None
if "reload_after_it" in training:
    reload_after_it = training["reload_after_it"]
    del training["reload_after_it"]
samples_per_it = None
if "samples_per_it" in training:
    samples_per_it = training["samples_per_it"]
    del training["samples_per_it"]
    if samples_per_it >= len(subjects):
        samples_per_it = None
        reload_after_it = None
if "fit_type" not in training:
    training["fit_type"] = 'custom'
    
    
num_samp = None
outer_num_its=1
    
if reload_after_it is not None:
    outer_num_its = training['num_its']//reload_after_it
    num_samp =  samples_per_it
    training['num_its'] = reload_after_it



#%

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


# mem check on GPU
mem_limit = 2500

if "CUDA_VISIBLE_DEVICES" in os.environ:
 
    mem = get_gpu_memory()
    if mem is None:
        print("seems that no is GPU present")            
    elif mem < mem_limit:
        raise NameError('Not enough memory on GPU, stopping!')
    else:
        print("Running on GPU, memfree: " + str(mem) + " MB")

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if len(gpus) > 0:
        tf.config.experimental.set_memory_growth(gpus[0],True)

else:

    print("CUDA_VISIBLE_DEVICES is not set")            

 

#%%

loading['nD'] = nD

def get_data(n=None):
   return patchwork.improc_utils.load_data_structured(
                       contrasts=contrasts,labels=labels,subjects=subjects,max_num_data=n, **loading)

print("\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>> loading first example for init.")

#with tf.device("/cpu:0"):    
#    tset,lset,rset,subjs = get_data(1)
    
if len(tset) == 0:
    print("Error: No data found!")
    raise NameError('No data found! Stopping ...  ')


#% load or generate model

if os.path.isfile(modelfi+".json") and not reinit_model:
    print("\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> model already existing, loading ")
    themodel = patchwork.PatchWorkModel.load(modelfi,immediate_init=True,notmpfile=True)
else:

    print("\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> creating new model")    
    
    patching['ndim'] = nD    
    network['modelname'] = modelfi  
    network['num_labels']= -1
    network['spatial_train']= False
    network['classifier_train']= True
    network['num_classes'] = 1
    print('numlabels:' + str(network['num_labels']))

    
    # create model
    cgen = patchwork.CropGenerator(**patching)
    themodel = patchwork.PatchWorkModel(cgen, **network)
    
    print("\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>> initializing network")
    dataexample = tset[0][0:1,...]    
   # cc = themodel.apply_full(dataexample,resolution=rset[0],repetitions=80,generate_type='random',verbose=True,
    #                         max_patching =True  )
# max_patching    
    
    print(cc.shape)

print("\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>> model summary")
themodel.summary()

    
if themodel.train_cycle is None:    
   themodel.train_cycle = 0
themodel.train_cycle += 1   
if "align_physical" in loading:
    themodel.align_physical = loading["align_physical"]


    
#%% start training    


training = {
   "num_patches":30,
   #"augment": {"dphi":0.2, "flip":[1,0] , "dscale":[0.1,0.1] },
   "epochs":50,
   "num_its":100,                
   "loss": tf.keras.losses.binary_crossentropy,
   }




print("\n\n\n\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> starting training")
import gc
for i in range(0,outer_num_its):
    
    print("\n========================================================= loading data =========================================================")
    mem = get_gpu_memory()
    if mem is not None:
        print("free GPU mem:" + str(mem) + "MB")

    #with tf.device("/cpu:0"):    
        # if "use_unlabeled_data" in loading and loading["use_unlabeled_data"]:
        #     tset,lset,cset,rset,subjs = get_data(num_samp)
        #     unlabeled_ids = list(tf.squeeze(tf.where(tf.concat(cset,0)==0)).numpy())
        # else:
        #     unlabeled_ids = []
        #     tset,lset,rset,subjs = get_data(num_samp)
        
    # lset = [tf.cast([[1,0]],tf.float32),tf.cast([[0,1]],tf.float32)]
    # lset = [tf.cast([[1]],tf.float32),
    #         tf.cast([[0]],tf.float32),
    #         tf.cast([[0]],tf.float32)]
    # ff = 0.001;
    # tset[0] =tset[0]*ff
    # tset[1] =tset[1]*ff+100
    # tset[2] =tset[2]*ff+100

    themodel.train(tset,lset,resolutions=rset,**training,
                   debug=True,
                   batch_size=10,
                   max_agglomerative=True,
                   verbose=2,inc_train_cycle=False,
                   valid_ids=valid_ids)
    
    with tf.device("/cpu:0"):  
        del tset
        del lset
        gc.collect()
        #tset = None
        #lset = None
    
    sys.stdout.flush()























