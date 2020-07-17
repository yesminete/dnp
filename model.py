# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# pip install tensorflow==2.1.0 matplotlib pillow opencv-python  nibabel


#%%

import numpy as np
import math
#from PIL import Image
#import cv2

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras.callbacks import History 

from timeit import default_timer as timer
from os import path
import nibabel as nib
import json
import warnings

from .crop_generator import *
from .improc_utils import *
from .customLayers import *

#%%###############################################################################################

## A CNN wrapper to allow easy layer references and stacking

class CNNblock(layers.Layer):
  def __init__(self,theLayers=None,name=None,verbose=False):
    super().__init__(name=name)
    
    self.verbose = verbose
    if theLayers is None:
        self.theLayers = {}
    else:
        self.theLayers = theLayers

  def add(self,*args):
      if len(args) > 1:
        l = [];
        for a in args:
           if hasattr(a,'dest') and a.dest is not None:
               l.append({'f':a,'dest':a.dest})
           else:
               l.append({'f':a})
        self.theLayers[l[0]['f'].name] =l
      else:     
        if isinstance(args[0],list):
            self.theLayers[args[0][0].name] = list(args[0])
        else:
            self.theLayers[args[0].name] = args[0]
            
          
   


  def call(self, inputs, alphas=None, training=False):
    
    def apply_fun(f,x):
                
        
        if len(x.shape) == nD+2:
            self.last_spatial_shape = x.shape        
        if self.verbose:
            print("  " + f.name , "input_shape: " ,  x.shape)
        if hasattr(f,'isBi') and f.isBi:
            if self.verbose:
                if alphas is not None:
                    print("  " + f.name , "(bi) input_shape: " ,  alphas.shape)
            return f(x,alphas,training=training)
        else:
            return f(x,training=training)
        
    
    x = inputs
    nD = len(inputs.shape)-2 
    cats = {}
    for l in self.theLayers:
        cats[l] = []

    if self.verbose:
        print("----------------------")
        if alphas is not None:
            print("with alphas: " , alphas.shape)

    for l in sorted(self.theLayers):
      
      if self.verbose:
          print("metalayer: " + l)
        
      # this gathers all inputs that might have been forwarded from other layers
      forwarded = cats[l]
      for r in forwarded:
        if x is None:
          x = r
        else:
          x = tf.concat([x,r],nD+1)


      if isinstance(self.theLayers[l],str) and self.theLayers[l][0:7]  == 'reshape':
          prodsz = np.prod(self.last_spatial_shape[1:-1])
          outfdim = int(self.theLayers[l][8:])
          self.theLayers[l] = [layers.Dense(prodsz*outfdim),
                               layers.Reshape(self.last_spatial_shape[1:-1] + [outfdim])]


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
                  res = apply_fun(f,res)
                  #res = f(res,training=training)
            else:
                res = apply_fun(fun,res)
                #res = fun(res,training=training)
              

            if 'dest' in d:
              dest = d['dest']
              if self.verbose:
                  print("          dest:"+dest)
              cats[dest].append(res)  
            else:            
              y = res
          x = y
        else:
          for f in a:
            #x = f(x,training=training)
            x = apply_fun(f,x)
      else:
        #x = a(x,training=training)
        x = apply_fun(a,x)

    return x



def createCNNBlockFromObj(obj,custom_objects=None):

  def tolayer(x):
      if isinstance(x,dict):
          if 'keras_layer' in x:
              return layers.deserialize({'class_name':x['name'],'config':x['keras_layer']},
                                        custom_objects=custom_objects)
          if 'CNNblock' in x:
              return createCNNBlockFromObj(x['CNNblock'],custom_objects=custom_objects)            
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
  if isinstance(theLayers,list):
      res = []
      for l in theLayers:
          res.append(createCNNBlockFromObj(l,custom_objects=custom_objects))
      return res
  else:
      return CNNblock(theLayers)
      

############################ The definition of the Patchwork model

class PatchWorkModel(Model):
  def __init__(self,cropper,

               blockCreator=None,
               num_labels=1,
               intermediate_out=0,
               intermediate_loss=False,
               block_out=None,
               
               spatial_train=True,
               spatial_max_train=False,
               finalBlock=None,

               classifierCreator=None,
               num_classes=1,
               cls_intermediate_out=2,
               cls_intermediate_loss=False,
               classifier_train=False,

               preprocCreator=None,

               forward_type='simple',
               trainloss_hist = None,
               validloss_hist = None,
               trained_epochs = 0,
               modelname = None
               ):
    super().__init__()
    self.preprocessor = []
    self.blocks = []
    self.classifiers = []
    self.finalBlock=finalBlock
    self.set_cropgen(cropper)
    self.forward_type = forward_type
    self.num_labels = num_labels
    self.num_classes = num_classes

    if trainloss_hist is None:
        trainloss_hist = {}
    if validloss_hist is None:
        validloss_hist = {}


    self.intermediate_loss = intermediate_loss
    self.intermediate_out = intermediate_out
    self.block_out = block_out
    self.cls_intermediate_loss = cls_intermediate_loss
    self.cls_intermediate_out = cls_intermediate_out
    self.num_classes=num_classes
    self.classifier_train=classifier_train
    self.spatial_train=spatial_train or spatial_max_train
    self.spatial_max_train=spatial_max_train

        
        
    self.trainloss_hist = trainloss_hist
    self.validloss_hist = validloss_hist
    self.trained_epochs = trained_epochs
    self.modelname = modelname
    
    if modelname is not None and path.exists(modelname + ".json"):
         warnings.warn(modelname + ".json already exists!! Are you sure you want to override?")
        
    
    if self.block_out is None:
        self.block_out = []
        for k in range(self.cropper.depth-1): 
          self.block_out.append(num_labels+intermediate_out)
        if self.spatial_train:
          self.block_out.append(num_labels)
        
    
    for k in range(self.cropper.depth-1): 
      self.blocks.append(blockCreator(level=k, outK=self.block_out[k]))
    if self.spatial_train :
      self.blocks.append(blockCreator(level=cropper.depth-1, outK=self.block_out[cropper.depth-1]))

    if preprocCreator is not None:
       for k in range(self.cropper.depth): 
           self.preprocessor.append(preprocCreator(level=k))
        
    if classifierCreator is not None:
       for k in range(self.cropper.depth): 
         clsfier = classifierCreator(level=k,outK=num_classes+cls_intermediate_out)
         if clsfier is None:
             break
         self.classifiers.append(clsfier)
        
        
  
  def serialize_(self):
    return   { 'forward_type':self.forward_type,

               'blocks':self.blocks,
               'intermediate_out':self.intermediate_out,
               'intermediate_loss':self.intermediate_loss,   
               'block_out':self.block_out,   
               'spatial_train':self.spatial_train,
               'num_labels':self.num_labels,

               'classifiers':self.classifiers,
               'cls_intermediate_out':self.cls_intermediate_out,
               'cls_intermediate_loss':self.cls_intermediate_loss,   
               'classifier_train':self.classifier_train,
               'num_classes':self.num_classes,
               
               'preprocessor':self.preprocessor,

               'cropper':self.cropper,
               'finalBlock':self.finalBlock,
               'trainloss_hist':self.trainloss_hist,
               'validloss_hist':self.validloss_hist,
               'trained_epochs':self.trained_epochs
            }


  def set_cropgen(self, cropper):
      self.cropper = cropper
      cropper.model = self



  def call(self, inputs, training=False, lazyEval=None, testIT=False):

    def subsel(inp,idx,w):
      sz = inp.shape
      nsz = [sz[0]/w,w]
      for k in range(len(sz)-1):
          nsz.append(sz[k+1])          
      inp = tf.reshape(inp,tf.cast(nsz,dtype=tf.int32))
      inp = tf.gather(inp,tf.squeeze(idx),axis=1)
      sz = inp.shape
      nsz = [sz[0]*sz[1]]
      for k in range(len(sz)-2):
          nsz.append(sz[k+2])                
      inp = tf.reshape(inp,tf.cast(nsz,dtype=tf.int32))
      return inp
  
    nD = self.cropper.ndim
    
    ## squeeze additional batch_dim if necearray
    original_shape = None
    if not callable(inputs) and len(inputs['input0'].shape) > nD+2:
        original_shape = inputs['input0'].shape[1]
        for k in inputs:
            sz = inputs[k].shape
            if nD == 2:            
                newsz = [-1, sz[2], sz[3], sz[4]]
            else:
                newsz = [-1, sz[2], sz[3], sz[4], sz[5]]
            inputs[k] = tf.reshape(inputs[k],newsz)

    
    output = []
    res = None
    res_nonspatial = None
    #################### main loop over depth
    for k in range(self.cropper.depth):



      # lazy Evaluation
      idx = None
      if k>0 and lazyEval is not None:
        reduceFun= lazyEval['reduceFun']
        attentionFun= lazyEval['attentionFun']
        fraction= lazyEval['fraction']                 
        if reduceFun == 'classifier_output':
            assert res_nonspatial is not None, "no classifier output for attention in lazyEval available!!"
            attention = res_nonspatial[:,0]
        else:
            if lazyEval['label'] is None:        
                attention = reduceFun(attentionFun(res[...,0:self.num_labels]),axis=list(range(1,nD+2)))                  
            else:
                attention = reduceFun(attentionFun(res[...,lazyEval['label']:lazyEval['label']+1]),axis=list(range(1,nD+2)))                  
        idx = tf.argsort(attention,0,'DESCENDING')
        numps = tf.cast(tf.floor(idx.shape[0]*fraction)+1,dtype=tf.int32)
        idx = idx[0:numps]            
        print('lazyEval, level ' + str(k-1) + ': only forwarding the ' + str(numps.numpy()) + ' most likely patches to next level')
        res = tf.gather(res,idx,axis=0)
        if res_nonspatial is not None:
           res_nonspatial = tf.gather(res_nonspatial,idx,axis=0)


      last = res

      ## get data and cropcoords at currnet scale
      if callable(inputs):
          inp,coords = inputs(idx=idx)          
      else:
          inp = inputs['input' + str(k)]
          coords = inputs['cropcoords' + str(k)]

      inp_nonspatial = res_nonspatial
      
           
      
      if k > 0: # it's not the initial scale
          
         # get result from last scale
         if last.shape[0] is not None and coords.shape[0] != last.shape[0]:         
            mtimes = coords.shape[0]//last.shape[0]
            multiples = [1]*(nD+2)
            multiples[0] = mtimes
            last = tf.tile(last,multiples)   
            if inp_nonspatial is not None:
                multiples = [1]*(2)
                multiples[0] = mtimes
                inp_nonspatial = tf.tile(inp_nonspatial,multiples)
        
         # crop the relevant rgion
         if self.cropper.interp_type == 'lin':
             last_cropped = tf.gather_nd(last,tf.cast(0.5+coords,dtype=tf.int32),batch_dims=1)
         else:
             last_cropped = tf.gather_nd(last,coords,batch_dims=1)
         
         
         if k < len(self.preprocessor) :
             inp = self.preprocessor[k](inp,training=training)
         
         # cat with total input
         if self.forward_type == 'simple':
             if 0:#testIT:
                inp = last_cropped
             else:
                inp = tf.concat([inp,last_cropped],(nD+1))
         elif self.forward_type == 'mult':
            multiples = [1]*(nD+2)
            multiples[nD+1] = last_cropped.shape[nD+1]
            inp_ = tf.tile(inp,multiples) * last_cropped
            inp = tf.concat([inp,inp_,last_cropped],nD+1)
         else: 
            assert 0,"unknown forward type"

      else:
         if k < len(self.preprocessor) :
             inp = self.preprocessor[k](inp,training=training)
          

      ## now, apply the network at the current scale 
      
      # the classifier part
      current_output = []
      if len(self.classifiers) > 0:
         if k < len(self.classifiers):
             res_nonspatial = self.classifiers[k](inp,res_nonspatial,training=training) 
         if k < len(self.classifiers) and self.classifier_train:
             current_output.append(res_nonspatial[:,0:self.num_classes])
      
      # the spatial/segmentation part
      if self.spatial_train or  k < self.cropper.depth-1:
          if testIT:
              res=inp
          else:
              res = self.blocks[k](inp,res_nonspatial,training=training)      
      if self.spatial_train:
          if testIT:
              current_output.append(res)
          else:
              current_output.append(res[...,0:self.num_labels])

      output = output + current_output
    
    if not testIT:
        if self.finalBlock is not None and self.spatial_train:
            lo = output.pop()
            if isinstance(self.finalBlock,list):
                for fb in self.finalBlock:
                    output.append(fb(lo))
            else:                    
                output.append(self.finalBlock(lo,training=training))

    ## undo the sequueze of potential batch_dim2 and reduce via max
    if original_shape is not None:
        for k in range(len(output)):
            sz = output[k].shape
            newsz = [-1,original_shape]
            for j in range(len(sz)-1):
                newsz.append(sz[j+1])
            output[k] = tf.reshape(output[k],newsz)
            output[k] = tf.reduce_max(output[k],axis=1)
    if self.spatial_max_train:
        for k in range(len(output)):
            output[k] = tf.reduce_max(output[k],axis=list(range(1,nD+1)))
            
    if not self.intermediate_loss:
      if self.spatial_train and self.classifier_train:
         return [output[-2], output[-1]]
      else:
         return [output[-1]]          
    else:
      return output
  

  def apply_full(self, data,
                 resolution=None,
                 level=-1,
                 generate_type='tree',
                 jitter=0.05,
                 jitter_border_fix = False,
                 overlap=0,
                 repetitions=5,           
                 dphi=0,
                 branch_factor=1,
                 scale_to_original=True,
                 verbose=False,
                 num_chunks=1,
                 patch_size_factor=1,
                 lazyEval = None,
                 max_patching=False,
                 patch_stats= False,
                 testIT=False
                 ):


     start_total = timer()

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
         
     if lazyEval is not None:
         if isinstance(lazyEval,float) or isinstance(lazyEval,int):             
             lazyEval = {
                 'fraction' : lazyEval
             }
         if 'reduceFun' not in lazyEval:
             lazyEval['reduceFun'] =  tf.reduce_mean
         if 'attentionFun' not in lazyEval:
             lazyEval['attentionFun'] = tf.math.sigmoid
         if 'fraction' not in lazyEval:
             lazyEval['fraction'] = 0.2
         if 'label' not in lazyEval:
             lazyEval['label'] = None

              
     pstats = [None]*len(level)

     for w in range(num_chunks):
         if w > 0:
             print('gathering more to get full coverage: ' + str(w) + "/" +  str(num_chunks))
         for i in range(repetitions):
             
            if lazyEval is None:             
                print(">>> sampling patches for testing")
                start = timer()
            x = self.cropper.sample(data,None,test=False,generate_type=generate_type,
                                    resolutions=resolution,
                                    jitter = jitter,
                                    jitter_border_fix = jitter_border_fix,
                                    overlap=overlap,
                                    num_patches=reps,
                                    dphi=dphi,
                                    branch_factor=branch_factor,
                                    patch_size_factor=patch_size_factor,
                                    lazyEval=lazyEval,
                                    verbose=verbose)
            data_ = x.getInputData()
            if lazyEval is None:             
                print(">>> time elapsed, sampling: " + str(timer() - start) )

            print(">>> applying network -------------------------------------------------")
            start = timer()
            # use predict only if batch_size is the same across patch levels
            if False: #(generate_type == 'random' or generate_type == 'tree_full') and lazyEval is None and branch_factor==1:
                r = self.predict(data_)
                if not isinstance(r,list):
                    r = [r]
            # otherwise use ordinary apply
            else:
                r = self(data_,lazyEval=lazyEval,testIT=testIT)
            print(">>> time elapsed, network application: " + str(timer() - start) )
                
            if max_patching:
                
                for k in level:
                    if self.spatial_train:
                        sz = r[k].shape
                        tmp = tf.reduce_max(r[k],axis=list(range(1,len(sz)-1)),keepdims=True)
                        r[k] = tf.tile(tmp,[1] + list(sz[1:-1]) + [1])
                    else:
                        if k == -1:
                            sz = data_['input' + str(self.cropper.depth-1)].shape
                        else:
                            sz = data_['input' + str(k)].shape
                        for i in range(nD):
                            r[k] = tf.expand_dims(r[k],1)
                        r[k] = tf.tile(r[k],[1] + list(sz[1:-1]) + [1])

            if patch_stats:
                for k in level:
                    sz = r[k].shape
                    tmp = tf.reduce_max(r[k],axis=list(range(1,len(sz)-1)))
                    stat = {'max': tf.reduce_max(r[k],axis=list(range(1,len(sz)-1))),
                                   'mean': tf.reduce_mean(r[k],axis=list(range(1,len(sz)-1))),
                                   'mean2': tf.reduce_mean(tf.math.pow(r[k],2),axis=list(range(1,len(sz)-1))),
                                   }
                    if pstats[k] is None:
                        pstats[k] = stat
                    else:
                        for t in pstats[k]:
                            pstats[k][t] = tf.concat([pstats[k][t],stat[t]],0)
                    
                    
            print(">>> stitching result")
            start = timer()
            if (self.spatial_train or max_patching) and not self.spatial_max_train:
                for k in level:            
                  a,b = x.stitchResult(r,k)
                  pred[k] += a
                  sumpred[k] += b                
            print(">>> time elapsed, stitching: " + str(timer() - start) )
                  
         if (np.amin(sumpred[-1])) > 0:
             break
              
     if (self.spatial_train  or max_patching)  and not self.spatial_max_train:
         res = zipper(pred,sumpred,lambda a,b : a/(b+0.0001))     
         sz = data.shape
         orig_shape = sz[1:(nD+1)]
         if scale_to_original:
             for k in level:
                res[k] = tf.squeeze(resizeNDlinear(tf.expand_dims(res[k],0),orig_shape,True,nD))                        
         if single:
           res = res[0]
     
         end = timer()
         print(">>> total time elapsed: " + str(end - start_total) )
         
         if patch_stats:
             if single:
                 pstats = pstats[0]
             return res,pstats
         else:
             return res
     
     return r[0]

  # for multi-contrast data fname is a list
  def apply_on_nifti(self,fname, ofname=None,
                 generate_type='tree',
                 overlap=0,
                 jitter=0.05,
                 jitter_border_fix=False,
                 repetitions=5,
                 branch_factor=1,
                 num_chunks=1,
                 scale_to_original=True,
                 scalevalue=None,
                 dphi=0,
                 along4dim=False,
                 align_physical=True,
                 patch_size_factor=1,
                 crop_fdim=None,
                 crop_sdim=None,
                 lazyEval = None):

      def crop_spatial(img):
         if crop_sdim is not None:
            if nD == 2:
                img = img[crop_sdim[0],...]
                img = img[:,crop_sdim[1],...]
            if nD == 3:
                img = img[crop_sdim[0],...]
                img = img[:,crop_sdim[1],...]
                img = img[:,:,crop_sdim[2],...]
         return img
      nD = self.cropper.ndim
      if not isinstance(fname,list):
          fname = [fname]
      ims = []
      for f in fname:          
          img1 = nib.load(f)        
          if align_physical:
              img1 = align_to_physical_coords(img1)
          resolution = img1.header['pixdim'][1:4]
          
          a = img1.get_fdata()
          
          if crop_fdim is not None:
             if len(a.shape) > nD:
                a = a[...,crop_fdim]

        
          a = crop_spatial(a)
      
          a = np.expand_dims(np.squeeze(a),0)
          if len(a.shape) < nD+2:
              a = np.expand_dims(a,nD+1)
          a = tf.convert_to_tensor(a,dtype=self.cropper.ftype)
          ims.append(a)
      a = tf.concat(ims,nD+1)
      
      
      if scalevalue is not None:
          a = a * scalevalue


      do_app = lambda x: self.apply_full(x,generate_type=generate_type,
                                jitter=jitter,
                                jitter_border_fix=jitter_border_fix,
                                overlap=overlap,                            
                                repetitions=repetitions,
                                num_chunks=num_chunks,
                                branch_factor=branch_factor,
                                dphi=dphi,
                                resolution = resolution,
                                lazyEval = lazyEval,
                                patch_size_factor=patch_size_factor,
                                verbose=True,
                                scale_to_original=scale_to_original)


      if along4dim:
          res = []
          for k in range(a.shape[nD+1]):
              res.append(tf.expand_dims(do_app(a[...,k:k+1]),nD))
          res = tf.concat(res,nD)             
          
      else:         
          res = do_app(a)
          
      res = res.numpy()
      if nD == 2:
          if len(res.shape) == 3:         
              res = np.reshape(res,[res.shape[0],res.shape[1],1,res.shape[2]])
          if along4dim:
              res = np.reshape(res,[res.shape[0],res.shape[1],1,res.shape[2],res.shape[3]])
          img1.header.set_data_shape(res.shape)
          
      maxi = tf.reduce_max(tf.abs(res))
      fac = 32000/maxi
      
      newaffine = img1.affine
      
      if not scale_to_original:
          sz = img1.header.get_data_shape()
          facs = [res.shape[0]/sz[0],res.shape[1]/sz[1],res.shape[2]/sz[2]]
          img1.header.set_data_shape(res.shape)
          newaffine = np.matmul(img1.affine,np.array([[1/facs[0],0,0,0],[0,1/facs[1],0,0],[0,0,1/facs[2],0],[0,0,0,1]]))
      
      
      img1.header.set_data_dtype('int16')          

      pred_nii = None

      if ofname is not None:
    
          if isinstance(ofname,list):
             if len(res.shape) == 5:
                  for s in range(len(ofname)):
                      res_ = res[:,:,:,:,s]
                      img1.header.set_data_shape(res_.shape)
                      pred_nii = nib.Nifti1Image(res_*fac, newaffine, img1.header)
                      pred_nii.header.set_slope_inter(1/fac,0.0000000)
                      pred_nii.header['cal_max'] = 1
                      pred_nii.header['cal_min'] = 0
                      pred_nii.header['glmax'] = 1
                      pred_nii.header['glmin'] = 0
                      if ofname is not None:
                          nib.save(pred_nii,ofname[s])
     
          else:
             pred_nii = nib.Nifti1Image(res*fac, newaffine, img1.header)
             pred_nii.header.set_slope_inter(1/fac,0.0000000)
             pred_nii.header['cal_max'] = 1
             pred_nii.header['cal_min'] = 0
             pred_nii.header['glmax'] = 1
             pred_nii.header['glmin'] = 0
             if ofname is not None:
                  nib.save(pred_nii,ofname)
              

            
      if pred_nii is not None:        
          return pred_nii,res;
      else:
          return res



  def save(self,fname):
     outname = fname + ".json"
     with open(outname,'w') as outfile:
         json.dump(self, outfile,cls=patchworkModelEncoder)
     
     outname = fname + ".tf"        
     self.save_weights(outname,save_format='tf')

  @staticmethod
  def load(name,custom_objects={},show_history=False):

    custom_objects = custom_layers
    # custom_objects['biConvolution'] = biConvolution
    # custom_objects['normalizedConvolution'] = normalizedConvolution
    # custom_objects['squeezeLayer'] = squeezeLayer

    fname = name + ".json"
    with open(fname) as f:
        x = json.load(f)

    cropper = CropGenerator(**x['cropper'])
    del x['cropper']

    blocks = x['blocks']
    blkCreator = lambda level,outK=0 : createCNNBlockFromObj(blocks[level],custom_objects=custom_objects)
    del x['blocks']

    clsCreator = None
    if 'classifiers' in x:
        classi = x['classifiers']
        del x['classifiers']
        if classi is not None and len(classi) > 0:
            clsCreator = lambda level,outK=0 : (createCNNBlockFromObj(classi[level],custom_objects=custom_objects) if level < len(classi) else None)

    preprocCreator = None
    if 'preprocessor' in x:
        pproc = x['preprocessor']
        del x['preprocessor']
        if pproc is not None and len(pproc) > 0:
            preprocCreator = lambda level,outK=0 : createCNNBlockFromObj(pproc[level],custom_objects=custom_objects)


    fb = x['finalBlock']
    del x['finalBlock']

    finalBlock = None
    if fb is not None:
        finalBlock = createCNNBlockFromObj(fb, custom_objects=custom_objects)
    
    
    model = PatchWorkModel(cropper, blkCreator,
                           classifierCreator=clsCreator, 
                           preprocCreator=preprocCreator,
                           finalBlock=finalBlock,**x)
    model.load_weights(name + ".tf")
    model.modelname = name
    
    if show_history:
        model.show_train_stat()

    
    return model
    
  def show_train_stat(self):

    import matplotlib.pyplot as plt
    
    if isinstance(self.trainloss_hist,list):
        
        l = len(self.trainloss_hist[0][1])
        cols = 'rbymck'
        for k  in range(l):        
            x = [ i for i, j in self.trainloss_hist ]
            y = [ j[k] for i, j in self.trainloss_hist ]
            if k==0:
                plt.semilogy(x,y,cols[k],label="total train loss ")
            else:
                plt.semilogy(x,y,cols[k],label="train loss " + str(k))
        if len(self.validloss_hist) > 0:
            x = [ i for i, j in self.validloss_hist ]
            y = [ j for i, j in self.validloss_hist ]
            plt.semilogy(x,y,'g',label="valid loss")
        plt.legend(fontsize=10)
        plt.grid()
        plt.title(self.modelname)
        plt.pause(0.001)
        
    else:
#%%
        loss_hist = self.trainloss_hist
        
        def plothist(loss_hist,txt):
            cols = 'rbymck'   
            cnt = 0
            for k in sorted(loss_hist):
                x = [ i for i, j in loss_hist[k] ]
                y = [ j for i, j in loss_hist[k] ]
                if txt == "":                     
                    plt.semilogy(x,y,cols[cnt],label=txt+k,marker='o', linestyle='dashed')
                else:
                    plt.semilogy(x,y,cols[cnt],label=txt+k)
                cnt+=1
        plothist(self.trainloss_hist,'train_')
        plothist(self.validloss_hist,'')
        plt.legend()
        plt.grid()
        plt.title(self.modelname)
        plt.pause(0.001)  
            
#%%        


  # Trains the model for a certain number of iterations. For each iteration 
  # a new number of patches is sampled and fitted for a certain number of epochs.
  # Potentially an augmentation scheme can be applied,
  # input:
  #   trainset - your dataset
  #   labelset - your labelset     
  #   trainset.shape = [batch_dim,w,h,(d),f0]
  #   labelset.shape = [batch_dim,w,h,(d),f1]
  #     trainset and label set may also be list of tensors, which is needed
  #     when  dimensions of examples differ. But note that dimensions are 
  #     only allowed to differ, if init_scale=-1 or init_scale is set to 
  #     a certain fixed shape.
  #   num_its - number of iterations trained
  #   epochs - number of epochs trained in each iteration
  #   train_type - type of patch sampling scheme, 'random' or 'tree'
  #   num_patches - number of patch samples (random) or number of trees (tree).
  #   jitter,jitter_border_fix - if train_type=tree, the amount of randomness of
  #         tree branches (0<jitter<1). If jitter_border_fix=True, the border 
  #         patches are aligned with the border
  #   balance -  a dict {'ratio':r,'N':N,'numrounds':nr} , where r gives desired balance between
  #         positive and negative examples, N the number of tries per chunk and nr
  #         the number of chunks
  #   num_samples_per_epoch - either -1 (all) or a the number of samples taken from 
  #         trainset for patch generation (only possible if trainset is a list),
  #         num_samples_per_epoch <= len(trainset)
  #   valid_ids a list of indices in trainset, corresponding to vaidation examples
  #         not used for training
  #   showplot - show loss info during training
  #   autosave - save model automatlicaly during training
  #   augment  - a function for augmentation (data,labels) => (augdata,auglabels)
  # output:
  #   a list of levels (see createCropsLocal for content)



  def train(self,
            trainset,labelset, 
            resolutions=None,
            epochs=20, 
            num_its=100,
            traintype='random',
            num_patches=10,
            valid_ids = [],
            valid_num_patches=None,
            batch_size=None,
            verbose=1,
            steps_per_epoch=None,
            jitter=0,
            jitter_border_fix=False,
            balance=None,
            num_samples_per_epoch=-1,
            showplot=True,
            autosave=True,
            augment=None,
            max_agglomerative=False,
            rot_intrinsic=0,
            loss=None,
            optimizer=None
            ):
      
      
      
    def getSample(subset,valid=False):
        tset = [trainset[i] for i in subset]
        lset = [labelset[i] for i in subset]      
        rset = None
        
        dphi = rot_intrinsic
        if valid:
            dphi=0
        
        if resolutions is not None:
            rset = [resolutions[i] for i in subset]      
            
        if traintype == 'random':
            np = num_patches
            if valid:
                np = valid_num_patches
            c = self.cropper.sample(tset,lset,resolutions=rset,generate_type='random', 
                                    num_patches=np,augment=augment,balance=balance,dphi=dphi)
        elif traintype == 'tree':
            c = self.cropper.sample(tset,lset,resolutions=rset,generate_type='tree_full', jitter=jitter,
                                    jitter_border_fix=jitter_border_fix,augment=augment,balance=balance,dphi=dphi)
        return c
    
    
    if valid_num_patches is None:
        valid_num_patches = num_patches
      
    if loss is not None:
        if optimizer is None:
            optimizer = tf.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=True)
        print("compiling ...")
        self.compile(loss=loss, optimizer=optimizer)

              
            
    for i in range(num_its):
        print("======================================================================================")
        print("iteration:" + str(i))
        print("======================================================================================")

        trainidx = list(range(len(trainset)))
        trainidx = [item for item in trainidx if item not in valid_ids]
        
        ### sampling
        print("sampling patches for training")
        start = timer()
        if num_samples_per_epoch == -1:          
           subset = trainidx
        else:            
           subset = sample(trainidx,num_samples_per_epoch)
        c = getSample(subset)      

        # print(c.scales[0].keys())
        end = timer()
        print("time elapsed, sampling: " + str(end - start) )
      
        ### fitting
        history = History()
      
        b2dim = -1
        if max_agglomerative:
            b2dim = num_patches
        
        inputdata = c.getInputData(b2dim)
        targetdata = c.getTargetData(b2dim)
        print("starting training")
        start = timer()
        self.fit(inputdata,targetdata,
                  epochs=epochs,
                  verbose=verbose,
                  steps_per_epoch=steps_per_epoch,
                  batch_size=batch_size,
                  callbacks=[history])
        end = timer()
        
        c.scales=None
        c=None
        inputdata=None
        targetdata=None


        def accum_hist(loss_hist,cur_hist):
            for k in cur_hist:
                if k not in loss_hist:
                    loss_hist[k] = []                    
            for k in loss_hist:
                loss_hist[k] += list(zip(list(range(self.trained_epochs,self.trained_epochs+epochs)),cur_hist[k]))

        if isinstance( self.trainloss_hist,list):
            
            loss = [None]*len(history.history['loss'])
            for k in range(len(history.history['loss'])):
                loss[k] = []
                loss[k].append(history.history['loss'][k])           
                for j in range(5):
                    if ('output_'+str(j+1)+'_loss') not in history.history:
                        break
                    loss[k].append(history.history['output_'+str(j+1)+'_loss'][k])
            
            self.trainloss_hist = self.trainloss_hist + list(zip( list(range(self.trained_epochs,self.trained_epochs+epochs)),loss))
        else:
            accum_hist(self.trainloss_hist,history.history)
            
        
        print("time elapsed, fitting: " + str(end - start) )
        
        ### validation
        if len(valid_ids) > 0:
            print("sampling patches for validation")
            c = getSample(valid_ids,True)    
            print("validating")
            res = self.evaluate(c.getInputData(),c.getTargetData(),
                  verbose=2)                

            if isinstance(self.validloss_hist,list):
                self.validloss_hist = self.validloss_hist + [(self.trained_epochs,res)]
            else:    
                tmp = {}
                for k in range(len(res)):
                    tmp['validloss_' + str(k)] = [res[k]]
                accum_hist(self.validloss_hist,tmp)
            
            c = None
            res = None
            
        self.trained_epochs += epochs
            
            

        
        
        if autosave:
           if self.modelname is None:
               print("no name given, not able to save model!")
           else:
               self.save(self.modelname)

        if showplot:
            self.show_train_stat()
            
            
#%%



class patchworkModelEncoder(json.JSONEncoder):
    def default(self, obj):
        name = obj.__class__.__name__

        if name == "PatchWorkModel":
           return obj.serialize_()
        if name =="CropGenerator":
           return obj.serialize_()
        if name == "CNNblock":            
           return {'CNNblock': dict(obj.theLayers) }
        if isinstance(obj,layers.Layer):
           return {'keras_layer':obj.get_config(), 'name': obj.__class__.__name__ }
        if isinstance(obj,dict):
           return dict(obj)
        if isinstance(obj,list):
           return list(obj)
        if hasattr(obj,'tolist'):
           return obj.tolist()
        return json.dumps(obj,cls=patchworkModelEncoder)
        


def Augmenter( morph_width = 150,
                morph_strength=0.25,
                rotation_dphi=0.1,
                flip = None ,
                scaling = None,
                normal_noise=0,
                repetitions=1,
                include_original=True):
                

    def augment(data,labels):
        
            sz = data.shape
            if len(sz) == 4:
                nD = 2
            if len(sz) == 5:
                nD = 3
    
            if not include_original:
                data_res = []
                if labels is not None:
                    labels_res = []
            else:
                data_res = [data]
                if labels is not None:
                    labels_res = [labels]
    
            for k in range(repetitions):            
                if nD == 2:
                    X,Y = sampleDefField_2D(sz)                    
                    data_ = interp2lin(data,Y,X)        
                    if labels is not None:
                        labels_ = interp2lin(labels,Y,X)            
                if nD == 3:
                    X,Y,Z = sampleDefField_3D(sz)
                    data_ = interp3lin(data,X,Y,Z)        
                    if labels is not None:                
                        labels_ = interp3lin(labels,X,Y,Z)            
                
                if normal_noise > 0:
                    data_ = data_ + tf.random.normal(data_.shape, mean=0,stddev=normal_noise)
                
                if flip is not None:
                    for j in range(nD):
                        if flip[j]:
                            if np.random.uniform() > 0.5:
                                data_ = np.flip(data_,j+1)
                                if labels is not None:
                                    labels_ = np.flip(labels_,j+1)
                                
                
                
                data_res.append(data_)
                if labels is not None:            
                    labels_res.append(labels_)
            data_res = tf.concat(data_res,0)
            
            if labels is not None:            
                labels_res = tf.concat(labels_res,0)
                return data_res,labels_res
            else:            
                return data_res,None
            
    


    def sampleDefField_2D(sz):
        
        X,Y = np.meshgrid(np.arange(0,sz[2]),np.arange(0,sz[1]))
        
        phi = np.random.uniform(low=-rotation_dphi,high=rotation_dphi)
        
        wid = morph_width/4
        s = wid*wid*morph_strength
        dx = conv_gauss2D_fft(np.expand_dims(np.expand_dims(np.random.normal(0,1,X.shape),0),3),wid)
        dx = np.squeeze(dx)
        dy = conv_gauss2D_fft(np.expand_dims(np.expand_dims(np.random.normal(0,1,X.shape),0),3),wid)
        dy = np.squeeze(dy)
        
        scfacs = [1,1]
        if scaling is not None:
            if isinstance(scaling,list):
                scfacs = [1+np.random.uniform(-1,1)*scaling[0],1+np.random.uniform(-1,1)*scaling[1]]
            else:
                sciso = scaling*np.random.uniform(-1,1)
                scfacs = [1+sciso,1+sciso]
        
        
        cx = 0.5*sz[2]
        cy = 0.5*sz[1]
        nX = scfacs[0]*tf.math.cos(phi)*(X-cx) - scfacs[0]*tf.math.sin(phi)*(Y-cy) + cx + s*dx
        nY = scfacs[1]*tf.math.sin(phi)*(X-cx) + scfacs[1]*tf.math.cos(phi)*(Y-cy) + cy + s*dy
        #dY = np.random.normal(0,s,X.shape)
        
        return nX,nY        
    

    def sampleDefField_3D(sz):
        
        X,Y,Z = np.meshgrid(np.arange(0,sz[1]),np.arange(0,sz[2]),np.arange(0,sz[3]),indexing='ij')
        
        
        wid = morph_width/4
        s = wid*wid*morph_strength
        dx = conv_gauss3D_fft(np.expand_dims(np.expand_dims(np.random.normal(0,1,X.shape),0),4),wid)
        dx = np.squeeze(dx)
        dy = conv_gauss3D_fft(np.expand_dims(np.expand_dims(np.random.normal(0,1,X.shape),0),4),wid)
        dy = np.squeeze(dy)
        dz = conv_gauss3D_fft(np.expand_dims(np.expand_dims(np.random.normal(0,1,X.shape),0),4),wid)
        dz = np.squeeze(dz)
        
        
        scfacs = [1,1,1]
        if scaling is not None:
            if isinstance(scaling,list):
                scfacs = [1+np.random.uniform(-1,1)*scaling[0],1+np.random.uniform(-1,1)*scaling[1],1+np.random.uniform(-1,1)*scaling[2]]
            else:
                sciso = scaling*np.random.uniform(-1,1)
                scfacs = [1+sciso,1+sciso,1+sciso]
        
        
        cx = 0.5*sz[1]
        cy = 0.5*sz[2]
        cz = 0.5*sz[3]
        
        u, _, vh = np.linalg.svd(np.eye(3) + rotation_dphi*np.random.normal(0,1,[3,3]), full_matrices=True)
        R = np.dot(u[:, :6] , vh)
        
        dx = s*dx
        dy = s*dy
        dz = s*dz
        
        nX = scfacs[0]*R[0,0]*(X-cx) + scfacs[0]*R[0,1]*(Y-cy) + scfacs[0]*R[0,2]*(Z-cz) + cx + dx
        nY = scfacs[1]*R[1,0]*(X-cx) + scfacs[1]*R[1,1]*(Y-cy) + scfacs[1]*R[1,2]*(Z-cz) + cy + dy
        nZ = scfacs[2]*R[2,0]*(X-cx) + scfacs[2]*R[2,1]*(Y-cy) + scfacs[2]*R[2,2]*(Z-cz) + cz + dz
        
        return nX,nY,nZ
    

    return augment





