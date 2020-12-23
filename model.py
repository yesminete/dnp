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
import time

from random import sample 


from .crop_generator import *
from .improc_utils import *
from .customLayers import *

############################ The definition of the Patchwork model

class myHistory :
    
  def __init__(self,model):
      self.trainloss_hist = {}
      self.validloss_hist = {}
      self.model = model

  def accum(self,which,cur_hist,epochs):
      
      if self.trainloss_hist is None:
          self.trainloss_hist = {}
      if self.validloss_hist is None:
          self.validloss_hist = {}
          
      if which == 'train':
          loss_hist = self.trainloss_hist
      if which == 'valid':
          loss_hist = self.validloss_hist
      for k in cur_hist:
         if k not in loss_hist:
            loss_hist[k] = []                    
      for k in loss_hist:
         loss_hist[k] += list(zip(list(range(self.model.trained_epochs,self.model.trained_epochs+epochs)),cur_hist[k]))

    
  def show_train_stat(self):

    try: 
        
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
            plt.title(self.model.modelname)
            plt.pause(0.001)
            
        else:
    #%%
            loss_hist = self.trainloss_hist
            plt.cla()
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
            ps = os.path.split(self.model.modelname)
            tit = os.path.split(ps[0])[1] + "/" + ps[1];
            plt.title(tit)
            plt.pause(0.001)  
            plt.savefig(self.model.modelname + ".png")
    except:
        if 'DISPLAY' in os.environ:
            print("problems during plotting. Wrong Display?? (DISPLAY="+os.environ['DISPLAY']+")")
        else:                                                                                       
            print("problems during plotting. No DISPLAY set!")
            
    


class PatchWorkModel(Model):
  def __init__(self,cropper,

               blockCreator=None,
               num_labels=1,
               intermediate_out=0,
               intermediate_loss=False,
               identical_blocks=False,
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
               modelname = None,
               input_fdim = None
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

    self.myhist = myHistory(self)
    self.myhist.trainloss_hist = trainloss_hist
    self.myhist.validloss_hist = validloss_hist

    self.intermediate_loss = intermediate_loss
    self.intermediate_out = intermediate_out
    self.identical_blocks = identical_blocks
        
    self.block_out = block_out
    self.cls_intermediate_loss = cls_intermediate_loss
    self.cls_intermediate_out = cls_intermediate_out
    self.num_classes=num_classes
    self.classifier_train=classifier_train
    self.classifier_train_deprecated = False
    self.spatial_train=spatial_train or spatial_max_train
    self.spatial_max_train=spatial_max_train



    # for k in trainloss_hist:
    #     self.trainloss_hist[k] = []
    #     for j in range(len(trainloss_hist[k])):
    #        self.trainloss_hist[k].append((0,1))
    
    
    self.trained_epochs = trained_epochs
    self.modelname = modelname
    self.input_fdim = input_fdim
    
    
    
    if modelname is not None and path.exists(modelname + ".json"):
         warnings.warn(modelname + ".json already exists!! Are you sure you want to override?")
        
    
    if self.block_out is None:
        self.block_out = []
        for k in range(self.cropper.depth-1): 
          self.block_out.append(num_labels+intermediate_out)
        if self.spatial_train or self.classifier_train:
          self.block_out.append(num_labels)

    blkCreator = lambda level:  blockCreator(level=level,outK=self.block_out[level])

    import inspect
    signature = inspect.signature(blockCreator)
    if 'input_shape' in signature.parameters:
        blkCreator = lambda level:  blockCreator(level=level,outK=self.block_out[level],input_shape=cropper.get_patchsize(level))   
        
    if not identical_blocks:
        theBlockCreator = blkCreator
    else:
        btmp = [None]*2
        def theBlockCreator(level):
            level = level if level < 2 else 1
            if btmp[level] is None:
                btmp[level] = blkCreator(level)
            return btmp[level]
        
        
        
    for k in range(self.cropper.depth-1): 
      self.blocks.append(theBlockCreator(k))
    if self.spatial_train or self.classifier_train:
      self.blocks.append(theBlockCreator(cropper.depth-1))

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
               'identical_blocks':self.identical_blocks,   
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
               'trainloss_hist':self.myhist.trainloss_hist,
               'validloss_hist':self.myhist.validloss_hist,
               'trained_epochs':self.trained_epochs,
               'input_fdim':self.input_fdim
            }


  def set_cropgen(self, cropper):
      self.cropper = cropper
      cropper.model = self



  def call(self, inputs, training=False, lazyEval=None, stitch_immediate=False, testIT=False):

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
            if k != "dest_full_size" and k != 'parent_box_scatter_index':
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
         if k < len(self.classifiers) and self.classifier_train_deprecated:
             current_output.append(res_nonspatial[:,0:self.num_classes])
      
      # the spatial/segmentation part
      if self.spatial_train or  k < self.cropper.depth-1  or self.classifier_train:
          if testIT:
              res=inp
          else:
              res = self.blocks[k](inp,res_nonspatial,training=training)      
      if self.spatial_train or self.classifier_train:
          if testIT:
              current_output.append(res)
          else:
              current_output.append(res[...,0:self.num_labels])

      output = output + current_output
    
    if not testIT:
        if self.finalBlock is not None:
            lo = output.pop()
            if isinstance(self.finalBlock,list):
                for fb in self.finalBlock:
                    output.append(fb(lo))
            else:                    
                output.append(self.finalBlock(lo,training=training))

    ## undo the sequueze of potential batch_dim2 and reduce via max or stitch
    if original_shape is not None:
        
        if stitch_immediate> 1:                               
               stitched = stitchResult_withstride(output,-1,[inputs],'NN',stitch_immediate)
               output = [stitched]
        else:
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
      if self.spatial_train and self.classifier_train_deprecated:
         return [output[-2], output[-1]]
      else:
         return [output[-1]]          
    else:
      return output
  

  def apply_full(self, data,
                 resolution=None,
                 level=-1,
                 generate_type='random',
                 jitter=0.05,
                 jitter_border_fix = False,
                 overlap=0,
                 repetitions=1,           
                 dphi=0,
                 branch_factor=1,
                 scale_to_original=True,
                 verbose=False,
                 num_chunks=1,
                 patch_size_factor=1,
                 lazyEval = None,
                 max_patching=False,
                 patch_stats= False,
                 stitch_immediate=False,
                 testIT=False
                 ):


     start_total = timer()

     nD = self.cropper.ndim
     self.cropper.dest_full_size = [None]*self.cropper.depth

     
     self.input_fdim = data.shape[-1]

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
            
            if stitch_immediate == True:
                stitch_immediate = reps
            else:
                stitch_immediate = -1
            if stitch_immediate>1:
                data_ = x.getInputData([None,reps])
                data_['parent_box_scatter_index'] = x.scales[-1]['parent_box_scatter_index']
                data_['dest_full_size'] = x.scales[-1]['dest_full_size']
            else:
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
                r = self(data_,lazyEval=lazyEval,stitch_immediate=stitch_immediate,testIT=testIT)
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
                    
            if not stitch_immediate>1:
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
         if stitch_immediate>1:
             res = r
             res[0] = tf.reduce_mean(res[0],axis=0)
         else:
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

      def crop_spatial(img,c):
        if crop_sdim is not None:
            if c == None:
                c = crop_sdim
                if crop_sdim == 'minbox':
                    if nD == 2:
                        c = bbox2(img);
                    if nD == 3:
                        c = bbox3(img);
            if nD == 2:
                img = img[c[0],...]
                img = img[:,c[1],...]
            if nD == 3:
                img = img[c[0],...]
                img = img[:,c[1],...]
                img = img[:,:,c[2],...]
            return img,c
        else:
            return img,None
    
        
      scrop = None
     
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

        
          a,scrop = crop_spatial(a,scrop)
      
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
      if scrop is not None:
          offset = np.matmul(newaffine,(np.array([scrop[0][0],scrop[1][0],scrop[2][0],1])))
          newaffine[:,3] = offset
          
      
      if not scale_to_original:
          sz = a.shape[1:nD+1]
          facs = [res.shape[0]/sz[0],res.shape[1]/sz[1],res.shape[2]/sz[2]]
          img1.header.set_data_shape(res.shape)
          newaffine = np.matmul(img1.affine,np.array([[1/facs[0],0,0,0],[0,1/facs[1],0,0],[0,0,1/facs[2],0],[0,0,0,1]]))
      
      
      img1.header.set_data_dtype('int16')          

      pred_nii = None

      if ofname is not None:
    
          if isinstance(ofname,list):
             #if len(res.shape) == 5:
                  for s in range(len(ofname)):
                      res_ = res[...,s]
                      img1.header.set_data_shape(res_.shape)
                      pred_nii = nib.Nifti1Image(res_*fac, newaffine, img1.header)
                      pred_nii.header.set_slope_inter(1/(0.000001+fac),0.0000000)
                      pred_nii.header['cal_max'] = 1
                      pred_nii.header['cal_min'] = 0
                      pred_nii.header['glmax'] = 1
                      pred_nii.header['glmin'] = 0
                      if ofname is not None:
                          nib.save(pred_nii,ofname[s])
     
          else:
             pred_nii = nib.Nifti1Image(res*fac, newaffine, img1.header)
             pred_nii.header.set_slope_inter(1/(0.000001+fac),0.0000000)
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
  def load(name,custom_objects={},show_history=False,immediate_init=True,notmpfile=False):

    custom_objects = custom_layers

    name = name.replace(".json","")
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



    if immediate_init and model.input_fdim is not None: 

        if notmpfile:
            model.load_weights(name + ".tf")   
        else:
                
            import tempfile
            import shutil
            import glob
            import os
            
            tmpdir = tempfile.mkdtemp()
            for filename in glob.glob(name + ".*"):
                if os.path.isfile(filename):
                   shutil.copy(filename, tmpdir)
            head_tail = os.path.split(name) 
            loading_name_tmp = os.path.join(tmpdir, head_tail[1])
            model.load_weights(loading_name_tmp + ".tf")            
        
        
        try:
            if model.cropper.ndim == 3:
                initdat = tf.ones([1,32,32,32, model.input_fdim])
            else:
                initdat = tf.ones([1,32,32, model.input_fdim])    
            print("----------------- load/init network by minimal application")
            dummy = model.apply_full(initdat,resolution=[1,1,1],verbose=False,scale_to_original=False,generate_type='random',repetitions=1)        
            print("----------------- model and weights loaded")
        except:
            if not notmpfile:
                shutil.rmtree(tmpdir)
            raise 
        
        if not notmpfile:
            shutil.rmtree(tmpdir)
    else:        
        model.load_weights(name + ".tf")   


    model.modelname = name
    
    if show_history:
        model.show_train_stat()

    
    return model
    
  def show_train_stat(self):
      self.myhist.show_train_stat()
            
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
            sample_cache=None,
            rot_intrinsic=0,
            loss=None,
            optimizer=None,
            patch_on_cpu=True,
            callback=None
            ):
      
    if patch_on_cpu:
        DEVCPU = "/cpu:0"
    else:
        DEVCPU = "/gpu:0"
        

    def getSample(subset,valid=False):
        tset = [trainset[i] for i in subset]
        lset = [labelset[i] for i in subset]      
        rset = None
        
        dphi = rot_intrinsic
        if valid:
            dphi=0
        
        self.cropper.dest_full_size = [None]*self.cropper.depth

        
        if resolutions is not None:
            rset = [resolutions[i] for i in subset]      
            
        with tf.device(DEVCPU):    
    
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

    trainidx = list(range(len(trainset)))
    trainidx = [item for item in trainidx if item not in valid_ids]

    if sample_cache is not None:
        print("sampling patches for training")
        start = timer()
        c = getSample(trainidx)      
        end = timer()
        print("time elapsed, sampling: " + str(end - start) + " (for " + str(len(trainidx)*num_patches) + ")")
                     
            
    for i in range(num_its):
        print("----------------------------------------- iteration:" + str(i))
        
        if sample_cache is None:
            ### sampling
            print("sampling patches for training")
            start = timer()
            if num_samples_per_epoch == -1:          
               subset = trainidx
            else:            
               subset = sample(trainidx,num_samples_per_epoch)
            c = getSample(subset)      
            end = timer()
            print("time elapsed, sampling: " + str(end - start) + " (for " + str(len(trainidx)*num_patches) + ")")

        
        ### fitting
        history = History()
      
        sampletyp = [None, -1]
        if max_agglomerative:
            sampletyp[1] = num_patches
        if sample_cache is not None:
            sampletyp[0] = sample(range(len(trainidx)*num_patches),sample_cache)

        inputdata = c.getInputData(sampletyp)
        targetdata = c.getTargetData(sampletyp)
            
            
        print("starting training")
        start = timer()
        self.fit(inputdata,targetdata,
                  epochs=epochs,
                  verbose=verbose,
                  steps_per_epoch=steps_per_epoch,
                  batch_size=batch_size,
                  callbacks=[history])
        end = timer()
        
        if sample_cache is None:
            del c.scales
            del c
            del inputdata
            del targetdata
            # c.scales=None
            # c=None
            # inputdata=None
            # targetdata=None

        self.myhist.accum('train',history.history,epochs)


        # if isinstance( self.trainloss_hist,list):
            
        #     loss = [None]*len(history.history['loss'])
        #     for k in range(len(history.history['loss'])):
        #         loss[k] = []
        #         loss[k].append(history.history['loss'][k])           
        #         for j in range(5):
        #             if ('output_'+str(j+1)+'_loss') not in history.history:
        #                 break
        #             loss[k].append(history.history['output_'+str(j+1)+'_loss'][k])
            
        #     self.trainloss_hist = self.trainloss_hist + list(zip( list(range(self.trained_epochs,self.trained_epochs+epochs)),loss))
        # else:
        #     accum_hist(self.trainloss_hist,history.history)
            
        
        print("time elapsed, fitting: " + str(end - start) )
        self.trained_epochs += epochs
        
        ### validation
        if len(valid_ids) > 0:
            print("sampling patches for validation")
            c = getSample(valid_ids,True)    
            print("validating")
            res = self.evaluate(c.getInputData(),c.getTargetData(),
                  verbose=2)                

            if isinstance(self.myhist.validloss_hist,list):
                self.myhist.validloss_hist =self.myhist.validloss_hist + [(self.trained_epochs,res)]
            else:    
                tmp = {}
                for k in range(len(res)):
                    tmp['validloss_' + str(k)] = [res[k]]
                self.myhist.accum('valid',tmp,epochs)
            
            c = None
            res = None
            
            
        if callback is not None:
            callback(i)

        
        
        if autosave:
           if self.modelname is None:
               print("no name given, not able to save model!")
           else:
               self.save(self.modelname)

        if showplot:
            self.myhist.show_train_stat()
            
            
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
        
