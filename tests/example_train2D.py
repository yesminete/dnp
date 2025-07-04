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


import numpy as np
import tensorflow as tf
import math
from tensorflow.keras import Model
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import multiprocessing

    

#sys.path.append("/home/reisertm")
sys.path.append("/software")
import patchwork_master.patchwork as patchwork

#%%
if __name__ != '__main__':
    print("CUDA_VISIBLE_DEVICES=-1")
    os.environ['CUDA_VISIBLE_DEVICES'] = "-1"
else:
    
    try:    
        multiprocessing.set_start_method('forkserver')
    except:
        pass
    
    # define your data sources 
    contrasts = [ { 'subj1' :  'example2d.nii.gz',  
                    'subj2' :  'example2d_b.nii.gz' 
                    } ]
    labels   = [  { 'subj1' :  'example2d_label.nii.gz', 
                   'subj2' :  'example2d_label_b.nii.gz', 
                     } ]
    
    subjects = [ 'subj1'
                #, 'subj2'
                ];
    
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
        "depth":3,                    
        "scheme":{ 
            "patch_size":[32,32],                
            "destvox_mm": None,
            "destvox_rel":[3,3],
            "fov_mm":None,
            "fov_rel":[0.9,0.9],
         },
        "smoothfac_data" : 0,   
        "smoothfac_label" : 0, 
       # "categorial_label" :[2],
        "categorial_label" :None,
        "interp_type" : "NN",    
        "scatter_type" : "NN",
        "normalize_input" : 'mean',
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
            patchwork.customLayers.createUnet_v2(depth=5,outK=outK,nD=nD,input_shape=input_shape,feature_dim=[8,16,16,32,64]),
    #    "preprocCreator": lambda level: patchwork.customLayers.HistoMaker(trainable=True,init='ct',dropout=0,nD=nD,normalize=False),    
        "intermediate_out":8,
        "intermediate_loss":True,          
        "forward_type":"bridge",

        'space_loss':{'full':True},
        'finalBlock':patchwork.customLayers.identity(),
        
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
        "one_hot_index_list":None,
        "exclude_incomplete_labels":1
       # "integer_labels":True
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
       "num_patches":100,
     #  "augment": {"dphi":0.2, "flip":[0,0] , "dscale":[0.1,0.1] },
       "epochs":20,
       "num_its":100,                
       "balance":None,#{"ratio":0.5,"autoweight":1},
       #"loss": patchwork.customLayers.TopK_loss2D(K="inf",mismatch_penalty=True),
       #"hard_mining":0.1,
       #"hard_mining_maxage":50,
       "reload_after_it":5,
       "samples_per_it":15,
       "parallel":False,
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
    mem_limit = -1 #2500
    
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
    
     
    
    #%
    
    loading['nD'] = nD
    
    def get_data(n=None):
       return patchwork.improc_utils.load_data_structured(
                           contrasts=contrasts,labels=labels,subjects=subjects,max_num_data=n, **loading)
    
       tset = dat[0]
       lset = dat[1]
       subj = dat[-1]
       
       fdim = tset[0].shape[-1]
       for k in range(1,len(tset)):
           if tset[k].shape[-1] != fdim:
               print("warning: fdim of data inconsistent " + str(fdim) + " != " + str(tset[k].shape[-1]) + " for subject " + subj[k])
       
       fdim = lset[0].shape[-1]
       for k in range(1,len(lset)):
           if lset[k].shape[-1] != fdim:
               print("warning: fdim of labels inconsistent " + str(fdim) + " != " + str(lset[k].shape[-1]) + " for subject " + subj[k])
    
        
       return dat
    
    print("\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>> loading first example for init.")
    
    with tf.device("/cpu:0"):    
        tset,lset,rset,subjs = get_data(1)
        
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
        if 'finalBlock' not in network:
            network['finalBlock']=layers.Activation('sigmoid')
        network['modelname'] = modelfi  
        if patching['categorial_label'] is not None:
            network['num_labels']= len(patching['categorial_label'])
        else:
            network['num_labels']= lset[0].shape[nD+1]
            
        #network['num_labels'] = 2
            
        print('numlabels:' + str(network['num_labels']))
    
        
        
        # create model
        cgen = patchwork.CropGenerator(**patching)
        themodel = patchwork.PatchWorkModel(cgen, **network)
        
        print("\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>> initializing network")
        dataexample = tset[0][0:1,...]    
        themodel.apply_full(dataexample,resolution=rset[0],repetitions=100,generate_type='random',verbose=True)
        
    
    print("\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>> model summary")
    themodel.summary()
    
        
    if themodel.train_cycle is None:    
       themodel.train_cycle = 0
    themodel.train_cycle += 1   
    if "align_physical" in loading:
        themodel.align_physical = loading["align_physical"]
    
    
        
    #%% start training    
        
    # initial_learning_rate = 0.1
    # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    #     initial_learning_rate,
    #     decay_steps=10,
    #     decay_rate=0.96,
    #     staircase=True)    
    # training['optimizer'] = tf.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.9, beta_2=0.999, amsgrad=True)
    
    def tocateg(l):
        r = tf.argmax(l,axis=-1)+1
        r = tf.where(tf.reduce_sum(l,-1)==0,0,r)
        return tf.expand_dims(r,-1)
    
    
    print("\n\n\n\n>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> starting training")
    import gc
    for i in range(0,outer_num_its):
        
        print("\n========================================================= loading data =========================================================")
        mem = get_gpu_memory()
        if mem is not None:
            print("free GPU mem:" + str(mem) + "MB")
    
        with tf.device("/cpu:0"):    
            if "use_unlabeled_data" in loading and loading["use_unlabeled_data"]:
                tset,lset,cset,rset,subjs = get_data(num_samp)
                unlabeled_ids = list(tf.squeeze(tf.where(tf.concat(cset,0)==0)).numpy())
            else:
                unlabeled_ids = []
                tset,lset,rset,subjs = get_data(num_samp)
    
        # some cathegorals for testing        
        if i==0 and patching['categorial_label'] is not None:            
            lset[0] = tocateg(lset[0])
            lset[1] = tocateg(lset[1])
    
    
      
       # lset = [tf.cast([0],tf.float32),tf.cast([1],tf.float32)]
    
            
        lazyTrain={"fraction":0.5,"branch_factor":2,"label":None}
        lazyTrain['reduceFun'] =  tf.reduce_max             
        lazyTrain['attentionFun'] = tf.math.sigmoid
            
        lset=[]
        themodel.train(tset,lset,resolutions=rset,**training,
                       debug=True,              
                       patch_on_cpu=False,
                   #    loss=loss,
                       dontcare=False,
                       recompile_loss_optim=True,
                     #  lazyTrain=lazyTrain,
                       
                       
                  #     hard_mining=0.2,
                  #     hard_mining_order='balance',
                       verbose=2,inc_train_cycle=False,
                       valid_ids=valid_ids)
        
        with tf.device("/cpu:0"):  
            del tset
            del lset
            gc.collect()
            #tset = None
            #lset = None
        
        sys.stdout.flush()
    gewinner
    
    
    #%%
    
    import nibabel as nib
    import numpy as np
    #img = nib.load('example2d_b.nii.gz')
    #a = np.expand_dims(np.expand_dims(np.squeeze(img.get_fdata()),0),3)
    #a = tf.convert_to_tensor(a,dtype=tf.float32) 
    #resol = {"voxsize":img.header['pixdim'][1:4],"input_edges":img.affine}
    a = tset[0]
    resol = rset[0]
    res =     themodel.apply_full(a,resolution=resol,
                                  num_patches=500,num_chunks=1,
                                      augment={},
                                      generate_type='random',
                                      scale_to_original=False,
                                  #    patch_stats=True,
                          #            window='cos2',
     #                                 verbose=2,
     #level=2,
                         #      testIT=2,
                                  #    lazyEval={'fraction':0.3},
                                      branch_factor=1
                                      )
    
    #print(stats['max'])
  #  plt.imshow(res,vmax=20)
    print(tf.reduce_max(res))
    plt.imshow(res[:,:,7])
    #%%
    p = [40,120]
    a = res[p[0],p[1],:]
    print(a)
    
    r = tf.reduce_sum(res*a+themodel.offset*100,-1)
    plt.imshow(r)
    
    
    
    #%% 
    themodel = patchwork.PatchWorkModel.load(themodel.modelname)
    
    res =     themodel.apply_on_nifti('example2d.nii.gz','xxx.nii',out_typ='mask',repetitions=50,num_chunks=10,
                                      generate_type='random',
                                      lazyEval={'fraction':1}
                                      )
    
    
    
    plt.imshow(res[1][:,:,0,0])
    
    
    
    
    
    
    
    
    
    
    
    
    
