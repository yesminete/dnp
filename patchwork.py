# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# pip install tensorflow==2.1.0 matplotlib pillow opencv-python  nibabel


#%%

import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from tensorflow.keras import Model

from tensorflow.keras import layers
import nibabel as nib

import json

from improc_utils import *

from crop_generator import *




#%%###############################################################################################

## A CNN wrapper to allow easy layer references and stacking

class CNNblock(layers.Layer):
  def __init__(self,theLayers):
    super(CNNblock, self).__init__()
    self.theLayers = theLayers

  def call(self, inputs, training=False):
    x = inputs
    nD = len(inputs.shape)-2 
    cats = {}
    for l in self.theLayers:
        cats[l] = []

    
    for l in sorted(self.theLayers):

        
      # this gathers all inputs that might have been forwarded from other layers
      forwarded = cats[l]
      for r in forwarded:
        if x is None:
          x = r
        else:
          x = tf.concat([x,r],nD+1)


      a = self.theLayers[l]      
      if isinstance(a,list):
        if isinstance(a[0],dict):
          y = None
          for j in range(0,len(a)):
            d = a[j]
            fun = d['f'] 
            res = x
            if isinstance(fun,list):
              for f in fun:
                  res = f(res,training=training)
            else:
              res = fun(res,training=training)

            if 'dest' in d:
              dest = d['dest']
              cats[dest].append(res)  
            else:            
              y = res
          x = y
        else:
          for f in a:
            x = f(x,training=training)
      else:
        x = a(x,training=training)

    return x



def createCNNBlockFromObj(obj,custom_objects=None):

  def tolayer(x):
      if isinstance(x,dict):
          if 'keras_layer' in x:
              return layers.deserialize({'class_name':x['name'],'config':x['keras_layer']},
                                        custom_objects=custom_objects)
          if 'CNNblock' in x:
              return CNNblock(x['CNNblock'])            
          for k in x:
              x[k] = tolayer(x[k])
          return x
      if isinstance(x,list):
          for k in range(len(x)):
              x[k] = tolayer(x[k])
          return x
          
      return x
                                  
  theLayers = tolayer(obj)
  if isinstance(theLayers,layers.Layer):
      return theLayers
  else:
      return CNNblock(theLayers)
      



############################ The definition of the Patchwork model

class PatchWorkModel(Model):
  def __init__(self,cropper,
               blockCreator,
               forward_type='simple',
               num_labels=1,
               intermediate_loss=False,
               intermediate_out=0,
               finalBlock=None
               ):
    super(PatchWorkModel, self).__init__()
    self.blocks = []
    self.cropper = cropper
    cropper.model = self
    self.forward_type = forward_type
    self.num_labels = num_labels
    self.intermediate_loss = intermediate_loss
    self.intermediate_out = intermediate_out
    self.finalBlock=finalBlock
    
    if callable(blockCreator):
        for k in range(self.cropper.depth-1): 
          self.blocks.append(blockCreator(level=k, outK=num_labels+intermediate_out))
        self.blocks.append(blockCreator(level=cropper.depth-1, outK=num_labels))

  def call(self, inputs, training=False):
    nD = self.cropper.ndim
    output = []
    for k in range(self.cropper.depth):
      ## get data and cropcoords at currnet scale
      inp = inputs['input' + str(k)]
      coords = inputs['cropcoords' + str(k)]

      if len(output) > 0:
         # get result from last scale
         last = output[-1]
         if last.shape[0] is not None and coords.shape[0] != last.shape[0]:            
            multiples = [1]*(nD+2)
            multiples[0] = coords.shape[0]//last.shape[0]
            last = tf.tile(last,multiples)         
         last_cropped = tf.gather_nd(last,coords,batch_dims=1)
         
         # cat with input
         if self.forward_type == 'simple':
            inp = tf.concat([inp,last_cropped],(nD+1))
         elif self.forward_type == 'mult':
            multiples = [1]*(nD+2)
            multiples[nD+1] = last_cropped.shape[nD+1]
            inp_ = tf.tile(inp,multiples) * last_cropped
            inp = tf.concat([inp,inp_,last_cropped],nD+1)
         else: 
            assert "unknown forward type"

      ## apply the network at the current scale 
      res = self.blocks[k](inp)
      if nD == 2:
          output.append(res[:,:,:,0:self.num_labels])
      elif nD == 3:
          output.append(res[:,:,:,:,0:self.num_labels])
      else:
          assert "ahhhh"
    
    if self.finalBlock is not None:
        output[-1] = self.finalBlock(output[-1])
    
    if not self.intermediate_loss:
      return [output[-1]]
    else:
      return output
  
  # for multi-contrast data fname is a list
  def apply_on_nifti(self,fname, ofname=None,
                 generate_type='tree',
                 jitter=0.05,
                 repetitions=5):
      nD = self.cropper.ndim
      if not isinstance(fname,list):
          fname = [fname]
      ims = []
      for f in fname:          
          img1 = nib.load(f)        
          a = np.expand_dims(np.squeeze(img1.get_fdata()),0)
          if len(a.shape) < nD+2:
              a = np.expand_dims(a,nD+1)
          a = tf.convert_to_tensor(a,dtype=tf.float32)
          ims.append(a)
      a = tf.concat(ims,nD+1)
          
      res = self.apply_full(a,generate_type=generate_type,
                            jitter=jitter,
                            repetitions=repetitions,
                            scale_to_original=True)

      pred_nii = nib.Nifti1Image(res, img1.affine, img1.header)
      if ofname is not None:
          nib.save(pred_nii,ofname)
      return pred_nii,res;


  def apply_full(self, data,
                 level=-1,
                 generate_type='tree',
                 jitter=0.05,
                 repetitions=5,
                 scale_to_original=False,
                 verbose=False
                 ):

     nD = self.cropper.ndim

     zipper = lambda a,b,f : list(map(lambda pair: f(pair[0],pair[1]) , list(zip(a, b))))

     single = False
     if not isinstance(level,list):
       level = [level]
       single = True
     
     pred = [0] * len(level)
     sumpred = [0] * len(level)
     
     reps = 1
     if generate_type == 'random':
         reps = repetitions
         repetitions = 1         
     
     for i in range(repetitions):
        x = self.cropper.sample(data,None,test=False,generate_type=generate_type,
                                randfun = lambda s : tf.random.normal(s,stddev=jitter),
                                 num_patches=reps,verbose=verbose)
        data_ = x.getInputData()
        r = self(data_)
        for k in level:
          a,b = x.stitchResult(r,k)
          pred[k] += a
          sumpred[k] += b         
     res = zipper(pred,sumpred,lambda a,b : a/(b+0.0001))
        
     sz = data.shape
     orig_shape = sz[1:(nD+1)]
     if scale_to_original:
         for k in level:
            res[k] = tf.squeeze(self.cropper.resize(tf.expand_dims(res[k],0),orig_shape,True))
           
     
     if single:
       res = res[0]
     
     return res

  def save(self,fname):
     outname = fname + ".json"
     with open(outname,'w') as outfile:
         json.dump(self, outfile,cls=patchworkModelEncoder)
     
     outname = fname + ".tf"        
     self.save_weights(outname,save_format='tf')

  @staticmethod
  def load(name,custom_objects=None):
    fname = name + ".json"
    with open(fname) as f:
        x = json.load(f)
    cropper = CropGenerator(**x['cropper'])
    del x['cropper']
    blocks = x['blocks']
    blkCreator = lambda level,outK=0 : createCNNBlockFromObj(blocks[level]['CNNblock'],custom_objects=custom_objects)
    del x['blocks']
    fb = x['finalBlock']
    del x['finalBlock']

    finalBlock = None
    if fb is not None:
        finalBlock = createCNNBlockFromObj(fb, custom_objects=custom_objects)
    
    
    model = PatchWorkModel(cropper, blkCreator,finalBlock=finalBlock,**x)
    model.load_weights(name + ".tf")
    return model
    


class patchworkModelEncoder(json.JSONEncoder):
    def default(self, obj):

        if isinstance(obj,PatchWorkModel):
           return { 'forward_type':obj.forward_type,
                    'num_labels':obj.num_labels,
                    'intermediate_out':obj.intermediate_out,
                    'intermediate_loss':obj.intermediate_loss,   
                    'blocks':obj.blocks,
                    'cropper':obj.cropper,
                    'finalBlock':obj.finalBlock
                 }
        if isinstance(obj,CropGenerator):
           return { 'patch_size':obj.patch_size,
                    'scale_fac' :obj.scale_fac,
                    'init_scale':obj.init_scale,
                    'overlap':obj.overlap_perc,
                    'depth':obj.depth,
                    'ndim':obj.ndim
                 }
        if isinstance(obj,CNNblock):            
           return {'CNNblock': obj.theLayers }
        if isinstance(obj,layers.Layer):
           return {'keras_layer':obj.get_config(), 'name': obj.__class__.__name__ }
        if isinstance(obj,dict):
           return dict(obj)
        return json.dumps(obj,cls=patchworkModelEncoder)
        





