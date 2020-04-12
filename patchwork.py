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
from tensorflow.keras.callbacks import History 

from timeit import default_timer as timer

from os import path

import nibabel as nib

import json

import warnings
from crop_generator import *

from improc_utils import *



#%%###############################################################################################

## A CNN wrapper to allow easy layer references and stacking

class CNNblock(layers.Layer):
  def __init__(self,theLayers=None,name=None):
    super(CNNblock, self).__init__(name=name)
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
        if hasattr(f,'isBi') and f.isBi:
            return f(x,alphas,training=training)
        else:
            return f(x,training=training)
        
    
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
                  res = apply_fun(f,res)
                  #res = f(res,training=training)
            else:
                res = apply_fun(fun,res)
                #res = fun(res,training=training)
              

            if 'dest' in d:
              dest = d['dest']
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
      


############################ biConv


class biConvolution(layers.Layer):

  def __init__(self, out_n=7, ksize=3, padding='SAME',transpose=False,nD=2,strides=None,**kwargs):
      
      
    super(biConvolution, self).__init__(**kwargs)
    self.out_n = out_n
    self.ksize = ksize
    self.padding = padding
    self.transpose = transpose
    self.nD = nD
    self.initializer = tf.random_normal_initializer(0, 0.05)
    self.num_alpha = 0
    self.isBi=True
    
    if strides is None:
        if nD == 2:
            strides = (1,1)
        else: 
            strides = (1,1,1,1,1)

    self.strides = strides

  def get_config(self):

        config = super().get_config().copy()
        config.update(
        {
            'out_n': self.out_n,
            'ksize': self.ksize,
            'strides': self.strides,
            'padding': self.padding,
            'transpose': self.transpose,
            'nD': self.nD,
            
        } )
    
        return config    
              
  def call(self, image,alphas = None):

    shape_im = image.shape
 
    if not hasattr(self,"weight"):
        
        
        if self.transpose:
            self.N = shape_im[self.nD+1]
            self.M = self.out_n
        else:
            self.M = shape_im[self.nD+1]
            self.N = self.out_n
        
        
        if alphas is not None:
          shape_alpha = alphas.shape    
          self.num_alpha = shape_alpha[1]
          if self.nD == 2:
              weight_shape = (self.num_alpha,self.ksize,self.ksize,self.M,self.N)                        
          else:
              weight_shape = (self.num_alpha,self.ksize,self.ksize,self.ksize,self.M,self.N)
             
        else:
          if self.nD == 2:
              weight_shape = (self.ksize,self.ksize,self.M,self.N)
          else:
              weight_shape = (self.ksize,self.ksize,self.ksize,self.M,self.N)
            
        self.weight = self.add_weight(shape=weight_shape, 
                        initializer=self.initializer, trainable=True,name=self.name)
    else:
        if alphas is not None and self.num_alpha == 0:
            assert 0,"inconsitent usage"
        if alphas is None and self.num_alpha > 0:
            assert 0,"inconsitent usage"


    if self.transpose:
        im_shape = tf.shape(image)
        output_shape_transpose = [0]*(self.nD+2)
        output_shape_transpose[0] = im_shape[0]
        output_shape_transpose[-1] = self.M
        for k in range(self.nD):
            output_shape_transpose[k+1] = im_shape[k+1]*self.strides[k] 

        if self.nD == 2:
            conv = lambda *a,**kw: tf.nn.conv2d_transpose(*a,**kw,output_shape=output_shape_transpose)
        else:
            conv = lambda *a,**kw: tf.nn.conv3d_transpose(*a,**kw,output_shape=output_shape_transpose)
    else:
        if self.nD == 2:
            conv = tf.nn.conv2d
        else:
            conv = tf.nn.conv3d


    x = 0
    if self.num_alpha == 0:
        x = conv(image, self.weight, strides=self.strides, padding=self.padding)
        
    else:
        for k in range(self.num_alpha):
            kernel = self.weight[k,...]
            alpha = alphas[:,k:k+1]
            for j in range(self.nD):
                alpha = tf.expand_dims(alpha,j+2)
            c = conv(image, kernel, strides=self.strides, padding=self.padding)
            x = x+alpha*c
    
    return x



############################ The definition of the Patchwork model

class PatchWorkModel(Model):
  def __init__(self,cropper,

               blockCreator,
               num_labels=1,
               intermediate_out=0,
               intermediate_loss=False,
               spatial_train=True,
               finalBlock=None,

               classifierCreator=None,
               num_classes=1,
               cls_intermediate_out=0,
               cls_intermediate_loss=False,
               classifier_train=False,

               forward_type='simple',
               trainloss_hist = [],
               validloss_hist = [],
               trained_epochs = 0,
               modelname = None
               ):
    super(PatchWorkModel, self).__init__()
    self.blocks = []
    self.classifiers = []
    self.cropper = cropper
    cropper.model = self
    self.forward_type = forward_type
    self.num_labels = num_labels
    self.num_classes = num_classes
    self.intermediate_loss = intermediate_loss
    self.intermediate_out = intermediate_out
    self.cls_intermediate_loss = cls_intermediate_loss
    self.cls_intermediate_out = cls_intermediate_out
    self.num_classes=num_classes
    self.classifier_train=classifier_train
    self.spatial_train=spatial_train

    self.finalBlock=finalBlock
    
    self.trainloss_hist = trainloss_hist
    self.validloss_hist = validloss_hist
    self.trained_epochs = trained_epochs
    self.modelname = modelname
    
    if modelname is not None and path.exists(modelname + ".json"):
         warnings.warn(modelname + ".json already exists!! Are you sure you want to override?")
        
    
    if callable(blockCreator):
        for k in range(self.cropper.depth-1): 
          self.blocks.append(blockCreator(level=k, outK=num_labels+intermediate_out))
        if self.spatial_train:
          self.blocks.append(blockCreator(level=cropper.depth-1, outK=num_labels))
        
    if classifierCreator is not None:
       for k in range(self.cropper.depth-1): 
         self.classifiers.append(classifierCreator(level=k,outK=num_classes+cls_intermediate_out))
       if self.classifier_train:
         self.classifiers.append(classifierCreator(level=cropper.depth-1,outK=num_classes))
        
        
  
  def serialize_(self):
    return   { 'forward_type':self.forward_type,

               'blocks':self.blocks,
               'intermediate_out':self.intermediate_out,
               'intermediate_loss':self.intermediate_loss,   
               'spatial_train':self.spatial_train,
               'num_labels':self.num_labels,

               'classifiers':self.classifiers,
               'cls_intermediate_out':self.cls_intermediate_out,
               'cls_intermediate_loss':self.cls_intermediate_loss,   
               'classifier_train':self.classifier_train,
               'num_classes':self.num_classes,

               'cropper':self.cropper,
               'finalBlock':self.finalBlock,
               'trainloss_hist':self.trainloss_hist,
               'validloss_hist':self.validloss_hist,
               'trained_epochs':self.trained_epochs
            }

  def call(self, inputs, training=False):
    nD = self.cropper.ndim
    output = []
    res = None
    res_nonspatial = None
    for k in range(self.cropper.depth):
      
      ## get data and cropcoords at currnet scale
      inp = inputs['input' + str(k)]
      inp_nonspatial = res_nonspatial
      coords = inputs['cropcoords' + str(k)]
      
      
      if len(output) > 0: # is it's not the initial scale
          
         # get result from last scale
         last = res
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
         last_cropped = tf.gather_nd(last,coords,batch_dims=1)
         
         # cat with total input
         if self.forward_type == 'simple':
             # for testing: inp = last_cropped
             inp = tf.concat([inp,last_cropped],(nD+1))
         elif self.forward_type == 'mult':
            multiples = [1]*(nD+2)
            multiples[nD+1] = last_cropped.shape[nD+1]
            inp_ = tf.tile(inp,multiples) * last_cropped
            inp = tf.concat([inp,inp_,last_cropped],nD+1)
         else: 
            assert 0,"unknown forward type"

      ## now, apply the network at the current scale 
      
      # the classifier part
      current_output = []
      if len(self.classifiers) > 0:
         if self.classifier_train or k < self.cropper.depth-1:
             res_nonspatial = self.classifiers[k](inp,inp_nonspatial) 
         if self.classifier_train:
             current_output.append(res_nonspatial[:,0:self.num_classes])
      
      # the spatial/segmentation part
      if self.spatial_train or  k < self.cropper.depth-1:
          res = self.blocks[k](inp,inp_nonspatial)      # for testing: res = inp      
      if self.spatial_train:
          current_output.append(res[...,0:self.num_labels])

      output = output + current_output
    
    if self.finalBlock is not None and self.spatial_train:
       output[-1] = self.finalBlock(output[-1])
    
    if not self.intermediate_loss:
      if self.spatial_train and self.classifier_train:
         return [output[-2], output[-1]]
      else:
         return [output[-1]]          
    else:
      return output
  
  # for multi-contrast data fname is a list
  def apply_on_nifti(self,fname, ofname=None,
                 generate_type='tree',
                 overlap=0,
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
                            overlap=overlap,                            
                            repetitions=repetitions,
                            scale_to_original=True)

      pred_nii = nib.Nifti1Image(res.numpy()*64000, img1.affine, img1.header)
      pred_nii.header.set_slope_inter(1/64000,0.00000001)
      if ofname is not None:
          nib.save(pred_nii,ofname)
      return pred_nii,res;


  def apply_full(self, data,
                 level=-1,
                 generate_type='tree',
                 jitter=0.05,
                 overlap=0,
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
                                jitter = jitter,
                                overlap=overlap,
                                 num_patches=reps,verbose=verbose)
        data_ = x.getInputData()
        if generate_type == 'random' or generate_type == 'tree_full':
            r = self.predict(data_)
            if not isinstance(r,list):
                r = [r]
        else:
            r = self(data_)
       
        if self.spatial_train:
            for k in level:            
              a,b = x.stitchResult(r,k)
              pred[k] += a
              sumpred[k] += b         
              
     if self.spatial_train:
         res = zipper(pred,sumpred,lambda a,b : a/(b+0.0001))        
         sz = data.shape
         orig_shape = sz[1:(nD+1)]
         if scale_to_original:
             for k in level:
                res[k] = tf.squeeze(resizeNDlinear(tf.expand_dims(res[k],0),orig_shape,True,nD,edge_center=False))                        
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
  def load(name,custom_objects={},show_history=True):

    custom_objects['biConvolution'] = biConvolution

    fname = name + ".json"
    with open(fname) as f:
        x = json.load(f)

    cropper = CropGenerator(**x['cropper'])
    del x['cropper']

    blocks = x['blocks']
    blkCreator = lambda level,outK=0 : createCNNBlockFromObj(blocks[level]['CNNblock'],custom_objects=custom_objects)
    del x['blocks']

    clsCreator = None
    if 'classifiers' in x:
        classi = x['classifiers']
        del x['classifiers']
        if classi is not None and len(classi) > 0:
            clsCreator = lambda level,outK=0 : createCNNBlockFromObj(classi[level]['CNNblock'],custom_objects=custom_objects)

    fb = x['finalBlock']
    del x['finalBlock']

    finalBlock = None
    if fb is not None:
        finalBlock = createCNNBlockFromObj(fb, custom_objects=custom_objects)
    
    
    model = PatchWorkModel(cropper, blkCreator,classifierCreator=clsCreator, finalBlock=finalBlock,**x)
    model.load_weights(name + ".tf")
    model.modelname = name
    
    if show_history:
        model.show_train_stat()

    
    return model
    
  def show_train_stat(self):
    x = [ i for i, j in self.trainloss_hist ]
    y = [ j for i, j in self.trainloss_hist ]
    plt.semilogy(x,y,'r',label="train loss")
    if len(self.validloss_hist) > 0:
        x = [ i for i, j in self.validloss_hist ]
        y = [ j for i, j in self.validloss_hist ]
        plt.semilogy(x,y,'g',label="valid loss")
    plt.legend()
    plt.pause(0.001)


  def train(self,
            trainset,labelset, 
            epochs=20, 
            num_its=100,
            traintype='random',
            num_patches=10,
            valid_ids = [],
            jitter=0,
            num_samples_per_epoch=-1,
            showplot=True,
            autosave=True,
            augment=None
            ):
      
    def getSample(subset):
        tset = [trainset[i] for i in subset]
        lset = [labelset[i] for i in subset]      
        if traintype == 'random':
            c = self.cropper.sample(tset,lset,generate_type='random',  num_patches=num_patches,augment=augment)
        elif traintype == 'tree':
            c = self.cropper.sample(tset,lset,generate_type='tree_full', jitter=jitter,augment=augment)
        return c
      
    history = History()
            
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
      
        inputdata = c.getInputData()
        targetdata = c.getTargetData()
        print("starting training")
        start = timer()
        self.fit(inputdata,targetdata,
                  epochs=epochs,
                  verbose=2,
                  callbacks=[history])
        end = timer()
        
        self.trainloss_hist = self.trainloss_hist + list(zip( list(range(self.trained_epochs,self.trained_epochs+epochs)),history.history['loss']))
        self.trained_epochs += epochs
        print("time elapsed, fitting: " + str(end - start) )
        
        ### validation
        if len(valid_ids) > 0:
            print("sampling patches for validation")
            c = getSample(valid_ids)    
            print("validating")
            res = self.evaluate(c.getInputData(),c.getTargetData(),
                  verbose=2)                
            self.validloss_hist = self.validloss_hist + [(self.trained_epochs,res)]
            

        
        
        if autosave:
           if self.modelname is None:
               print("no name given, not able to save model!")
           else:
               self.save(self.modelname)

        if showplot:
            self.show_train_stat()

# #%%
#             x = [ i for i, j in model.trainloss_hist ]
#             y = [ j for i, j in model.trainloss_hist ]
#             plt.semilogy(x,y,'r',label="train loss")
#             x = [ i for i, j in model.validloss_hist ]
#             y = [ j for i, j in model.validloss_hist ]
#             plt.semilogy(x,y,'g')
#             plt.legend()
      
#%%



class patchworkModelEncoder(json.JSONEncoder):
    def default(self, obj):
        
        name = obj.__class__.__name__

        if name == "PatchWorkModel":
           return obj.serialize_()
        if name =="CropGenerator":
           return obj.serialize_()
        if name == "CNNblock":            
           return {'CNNblock': obj.theLayers }
        if isinstance(obj,layers.Layer):
           return {'keras_layer':obj.get_config(), 'name': obj.__class__.__name__ }
        if isinstance(obj,dict):
           return dict(obj)
        return json.dumps(obj,cls=patchworkModelEncoder)
        





def Augmenter( morph_width = 150,
                morph_strength=0.25,
                rotation_dphi=0.1,
                flip = None ,
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
            labels_res = []
        else:
            data_res = [data]
            labels_res = [labels]

        for k in range(repetitions):            
            if nD == 2:
                X,Y = sampleDefField_2D(sz)                    
                data_ = interp2lin(data,Y,X)        
                labels_ = interp2lin(labels,Y,X)            
            if nD == 3:
                X,Y,Z = sampleDefField_3D(sz)
                data_ = interp2lin(data,Z,Y,X)        
                labels_ = interp2lin(labels,Z,Y,X)            
            
            if normal_noise > 0:
                data_ = data_ + tf.random.normal(data_.shape, mean=0,stddev=normal_noise)
            
            data_res.append(data_)
            labels_res.append(labels_)
        data_res = tf.concat(data_res,0)
        labels_res = tf.concat(labels_res,0)
        
        return data_res,labels_res
        



    def sampleDefField_2D(sz):
        
        X,Y = np.meshgrid(np.arange(0,sz[2]),np.arange(0,sz[1]))
        
        phi = np.random.uniform(low=-rotation_dphi,high=rotation_dphi)
        
        wid = morph_width/4
        s = wid*wid*morph_strength
        dx = conv_gauss2D_fft(np.expand_dims(np.expand_dims(np.random.normal(0,1,X.shape),0),3),wid)
        dx = np.squeeze(dx)
        dy = conv_gauss2D_fft(np.expand_dims(np.expand_dims(np.random.normal(0,1,X.shape),0),3),wid)
        dy = np.squeeze(dy)
        
        cx = 0.5*sz[2]
        cy = 0.5*sz[1]
        nX = tf.math.cos(phi)*(X-cx) - tf.math.sin(phi)*(Y-cy) + cx + s*dx
        nY = tf.math.sin(phi)*(X-cx) + tf.math.cos(phi)*(Y-cy) + cy + s*dy
        #dY = np.random.normal(0,s,X.shape)
        
        return nX,nY        
    

    return augment





