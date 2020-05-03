# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# pip install tensorflow==2.1.0 matplotlib pillow opencv-python  nibabel


#%%

import numpy as np
import math
import matplotlib.pyplot as plt
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
  def __init__(self,theLayers=None,name=None):
    super().__init__(name=name)
    
    self.verbose = False
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
        if self.verbose:
            print(f.name , "input_shape: " ,  x.shape)
        if hasattr(f,'isBi') and f.isBi:
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
               cls_intermediate_out=2,
               cls_intermediate_loss=False,
               classifier_train=False,

               preprocCreator=None,

               forward_type='simple',
               trainloss_hist = [],
               validloss_hist = [],
               trained_epochs = 0,
               modelname = None
               ):
    super().__init__()
    self.preprocessor = []
    self.blocks = []
    self.classifiers = []
    self.finalBlock=finalBlock

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

    
    self.trainloss_hist = trainloss_hist
    self.validloss_hist = validloss_hist
    self.trained_epochs = trained_epochs
    self.modelname = modelname
    
    if modelname is not None and path.exists(modelname + ".json"):
         warnings.warn(modelname + ".json already exists!! Are you sure you want to override?")
        
    
    for k in range(self.cropper.depth-1): 
      self.blocks.append(blockCreator(level=k, outK=num_labels+intermediate_out))
    if self.spatial_train:
      self.blocks.append(blockCreator(level=cropper.depth-1, outK=num_labels))

    if preprocCreator is not None:
       for k in range(self.cropper.depth): 
           self.preprocessor.append(preprocCreator(level=k))
        
    if classifierCreator is not None:
       for k in range(self.cropper.depth-1): 
         clsfier = classifierCreator(level=k,outK=num_classes+cls_intermediate_out)
         if clsfier is None:
             break
         self.classifiers.append(clsfier)
       if self.classifier_train and not self.cropper.create_indicator_classlabels:
         clsfier = classifierCreator(level=cropper.depth-1,outK=num_classes)
         if clsfier is not None:
            self.classifiers.append(clsfier)
        
        
  
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
               
               'preprocessor':self.preprocessor,

               'cropper':self.cropper,
               'finalBlock':self.finalBlock,
               'trainloss_hist':self.trainloss_hist,
               'validloss_hist':self.validloss_hist,
               'trained_epochs':self.trained_epochs
            }




  def call(self, inputs, training=False, lazyEval=None):

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


    testIT=False
      
    nD = self.cropper.ndim
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
            attention = reduceFun(attentionFun(res[...,0:self.num_labels]),axis=list(range(1,nD+2)))                  
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
         last_cropped = tf.gather_nd(last,coords,batch_dims=1)
         
         
         if k < len(self.preprocessor) :
             inp = self.preprocessor[k](inp)
         
         # cat with total input
         if self.forward_type == 'simple':
             if testIT:
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
             inp = self.preprocessor[k](inp)
          

      ## now, apply the network at the current scale 
      
      # the classifier part
      current_output = []
      if len(self.classifiers) > 0:
         if k < len(self.classifiers):
             res_nonspatial = self.classifiers[k](inp,inp_nonspatial) 
         if k < len(self.classifiers) and self.classifier_train:
             current_output.append(res_nonspatial[:,0:self.num_classes])
      
      # the spatial/segmentation part
      if self.spatial_train or  k < self.cropper.depth-1:
          if testIT:
              res=inp
          else:
              res = self.blocks[k](inp,inp_nonspatial)      
      if self.spatial_train:
          current_output.append(res[...,0:self.num_labels])

      output = output + current_output
    
    if not testIT:
        if self.finalBlock is not None and self.spatial_train:
            output[-1] = self.finalBlock(output[-1])
    
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
                 branch_factor=1,
                 scale_to_original=True,
                 verbose=False,
                 num_chunks=1,
                 lazyEval = None
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
                                    branch_factor=branch_factor,
                                    lazyEval=lazyEval,
                                    verbose=verbose)
            data_ = x.getInputData()
            if lazyEval is None:             
                print(">>> time elapsed, sampling: " + str(timer() - start) )

            print(">>> applying network -------------------------------------------------")
            start = timer()
            if (generate_type == 'random' or generate_type == 'tree_full') and lazyEval is None:
                r = self.predict(data_)
                if not isinstance(r,list):
                    r = [r]
            else:
                r = self(data_,lazyEval=lazyEval)
            print(">>> time elapsed, network application: " + str(timer() - start) )
                
            print(">>> stitching result")
            start = timer()
            if self.spatial_train:
                for k in level:            
                  a,b = x.stitchResult(r,k)
                  pred[k] += a
                  sumpred[k] += b                
            print(">>> time elapsed, stitching: " + str(timer() - start) )
                  
         if (np.amin(sumpred[-1])) > 0:
             break
              
     if self.spatial_train:
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
     
     return res


  # for multi-contrast data fname is a list
  def apply_on_nifti(self,fname, ofname=None,
                 generate_type='tree',
                 overlap=0,
                 jitter=0.05,
                 jitter_border_fix=False,
                 repetitions=5,
                 branch_factor=1,
                 num_chunks=1,
                 scalevalue=None,
                 lazyEval = None):
      nD = self.cropper.ndim
      if not isinstance(fname,list):
          fname = [fname]
      ims = []
      for f in fname:          
          img1 = nib.load(f)        
          resolution = img1.header['pixdim'][1:4]
          a = np.expand_dims(np.squeeze(img1.get_fdata()),0)
          if len(a.shape) < nD+2:
              a = np.expand_dims(a,nD+1)
          a = tf.convert_to_tensor(a,dtype=self.cropper.ftype)
          ims.append(a)
      a = tf.concat(ims,nD+1)
      if scalevalue is not None:
          a = a * scalevalue

      res = self.apply_full(a,generate_type=generate_type,
                            jitter=jitter,
                            jitter_border_fix=jitter_border_fix,
                            overlap=overlap,                            
                            repetitions=repetitions,
                            num_chunks=num_chunks,
                            branch_factor=branch_factor,
                            resolution = resolution,
                            lazyEval = lazyEval,
                            verbose=True,
                            scale_to_original=True)

      res = res.numpy()
      if nD == 2:
          res = np.reshape(res,[res.shape[0],res.shape[1],1,res.shape[2]])
          img1.header.set_data_shape(res.shape)
          
      img1.header.set_data_dtype('uint16')          
      pred_nii = nib.Nifti1Image(res*64000, img1.affine, img1.header)
      pred_nii.header.set_slope_inter(1/64000,0.0000000)
      pred_nii.header['cal_max'] = 1
      pred_nii.header['cal_min'] = 0
      pred_nii.header['glmax'] = 1
      pred_nii.header['glmin'] = 0
      if ofname is not None:
          nib.save(pred_nii,ofname)
      return pred_nii,res;



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
    blkCreator = lambda level,outK=0 : createCNNBlockFromObj(blocks[level]['CNNblock'],custom_objects=custom_objects)
    del x['blocks']

    clsCreator = None
    if 'classifiers' in x:
        classi = x['classifiers']
        del x['classifiers']
        if classi is not None and len(classi) > 0:
            clsCreator = lambda level,outK=0 : (createCNNBlockFromObj(classi[level]['CNNblock'],custom_objects=custom_objects) if level < len(classi) else None)

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
    plt.legend()
    plt.pause(0.001)


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
            jitter=0,
            jitter_border_fix=False,
            balance=None,
            num_samples_per_epoch=-1,
            showplot=True,
            autosave=True,
            augment=None,
            loss=None,
            optimizer=None
            ):
      
    def getSample(subset):
        tset = [trainset[i] for i in subset]
        lset = [labelset[i] for i in subset]      
        rset = None
        if resolutions is not None:
            rset = [resolutions[i] for i in subset]      
            
        if traintype == 'random':
            c = self.cropper.sample(tset,lset,resolutions=rset,generate_type='random',  num_patches=num_patches,augment=augment,balance=balance)
        elif traintype == 'tree':
            c = self.cropper.sample(tset,lset,resolutions=rset,generate_type='tree_full', jitter=jitter,jitter_border_fix=jitter_border_fix,augment=augment,balance=balance)
        return c
      
    if loss is not None:
        if optimizer is None:
            optimizer = tf.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=True)
        print("compiling ...")
        self.compile(loss=loss, optimizer=optimizer)

              
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
        
        c.scales=None
        c=None
        inputdata=None
        targetdata=None
        
        
        loss = [None]*len(history.history['loss'])
        for k in range(len(history.history['loss'])):
            loss[k] = []
            loss[k].append(history.history['loss'][k])           
            for j in range(5):
                if ('output_'+str(j+1)+'_loss') not in history.history:
                    break
                loss[k].append(history.history['output_'+str(j+1)+'_loss'][k])

        
        self.trainloss_hist = self.trainloss_hist + list(zip( list(range(self.trained_epochs,self.trained_epochs+epochs)),loss))
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
        if hasattr(obj,'tolist'):
           return obj.tolist()
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
                    data_ = interp3lin(data,X,Y,Z)        
                    labels_ = interp3lin(labels,X,Y,Z)            
                
                if normal_noise > 0:
                    data_ = data_ + tf.random.normal(data_.shape, mean=0,stddev=normal_noise)
                
                if flip is not None:
                    for j in range(nD):
                        if flip[j]:
                            if np.random.uniform() > 0.5:
                                data_ = np.flip(data_,j+1)
                                labels_ = np.flip(labels_,j+1)
                                
                
                
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
    

    def sampleDefField_3D(sz):
        
        X,Y,Z = np.meshgrid(np.arange(0,sz[1]),np.arange(0,sz[2]),np.arange(0,sz[3]),indexing='ij')
        
        
        wid = morph_width/4
        s = wid*wid*morph_strength
        dx = conv_gauss3D_fft(np.expand_dims(np.expand_dims(np.random.normal(0,1,X.shape),0),4),wid)
        dx = np.squeeze(dx)
        dy = conv_gauss3D_fft(np.expand_dims(np.expand_dims(np.random.normal(0,1,X.shape),0),4),wid)
        dy = np.squeeze(dy)
        dz = conv_gauss3D_fft(np.expand_dims(np.expand_dims(np.random.normal(0,1,X.shape),0),4),wid)
        dz = np.squeeze(dy)
        
        cx = 0.5*sz[1]
        cy = 0.5*sz[2]
        cz = 0.5*sz[3]
        
        u, _, vh = np.linalg.svd(np.eye(3) + rotation_dphi*np.random.normal(0,1,[3,3]), full_matrices=True)
        R = np.dot(u[:, :6] , vh)
        
        dx = s*dx
        dy = s*dy
        dz = s*dz
        
        nX = R[0,0]*(X-cx) + R[0,1]*(Y-cy) + R[0,2]*(Z-cz) + cx + dx
        nY = R[1,0]*(X-cx) + R[1,1]*(Y-cy) + R[1,2]*(Z-cz) + cy + dy
        nZ = R[2,0]*(X-cx) + R[2,1]*(Y-cy) + R[2,2]*(Z-cz) + cz + dz
        
        return nX,nY,nZ
    

    return augment





