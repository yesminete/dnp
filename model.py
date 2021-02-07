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
import tensorflow.keras.backend as kb

from timeit import default_timer as timer
from os import path
import nibabel as nib
import json
import warnings
import time
import sys

from random import sample 


from .crop_generator import *
from .improc_utils import *
from .customLayers import *




class FilterOut(object):
    def __init__(self, filterstr, *args):
        self.handles = [sys.__stdout__]
        self.omitted = 0
        self.omitted_line = ""
        self.filterstr = filterstr
    def write(self, s):
        for f in self.handles:
            if self.filterstr in s:                
                self.omitted +=1
                self.omitted_line = s
            elif self.omitted > 0 and s == '\n':
                s
            elif self.omitted > 0:
                f.write(self.omitted_line+"\n")
                f.write('omitted ' + str(self.omitted) +' lines.\n')
                self.omitted = 0
                f.write(s)                
            else:
                f.write(s)

                
    def flush(self):
        for f in self.handles:
            f.flush()



############################ The definition of the Patchwork model

class myHistory :
    
  def __init__(self,model):
      self.trainloss_hist = {}
      self.validloss_hist = {}
      self.model = model

  def accum(self,which,cur_hist,epochs,tensors=False,mean=False):
      
      if self.trainloss_hist is None:
          self.trainloss_hist = {}
      if self.validloss_hist is None:
          self.validloss_hist = {}
          
      if mean:
          mhist = {}
          for k in cur_hist[0]:
              mhist[k] = 0
              cnt=0
              for j in range(len(cur_hist)):
                  if k in cur_hist[j]:
                     mhist[k] += cur_hist[j][k]
                     cnt += 1
              mhist[k] = [mhist[k]/cnt]
          cur_hist = mhist

          
      if tensors:
          for k in cur_hist:
              cur_hist[k] = list(map(lambda x: x.numpy(),cur_hist[k]))
          
      if which == 'train':
          loss_hist = self.trainloss_hist
      else: # if which == 'valid':
          loss_hist = self.validloss_hist
      for k in cur_hist:
         if k not in loss_hist:
            loss_hist[k] = []                    
      for k in loss_hist:
         if k in cur_hist:
             loss_hist[k] += list(zip(list(range(self.model.trained_epochs,self.model.trained_epochs+epochs)),cur_hist[k]))
             
     
      new_minimum = False
      for k in self.validloss_hist:
          h = self.validloss_hist[k]
          a = filter(lambda x: x[0] > self.model.trained_epochs-5,h)
          a = list(map(lambda a:a[1],a))
          if len(a) < 2:
              break
          if 'loss' in k:
              if a[-1] < min(a[0:-1]):
                  new_minimum = True
          # else:
          #     if a[-1] > max(a[0:-1]):
          #         new_minimum = True
          
             
      return cur_hist, new_minimum
    
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
            plt.legend(fontsize=10,loc=3)
            plt.grid()
            plt.title(self.model.modelname)
            plt.pause(0.001)
            
        else:
            #%%
            loss_hist = self.trainloss_hist
            
            def plothist(loss_hist,txt):
                import matplotlib.pyplot as plt
                
                cols = 'rbymckrbymck'   
                cnt = 0
                for k in sorted(loss_hist):
                    x = [ i for i, j in loss_hist[k] ]
                    y = [ j for i, j in loss_hist[k] ]
                    n = int(np.ceil(len(y)/20.0))
                    y = np.convolve(y,np.ones([n])/n,mode='valid')
                    if n > 1:
                        y = np.concatenate([np.ones([n-1])*y[0],y],0)
                    
                    labeltxt = (txt+k).replace("_output","") + (": {:.2f}").format(math.log(y[-1])/math.log(10.0)   )
                    
                    
                    if txt == "":                     
                        plt.semilogy(x,y,cols[cnt],label=labeltxt,marker='o', linestyle='dashed')
                    else:
                        plt.semilogy(x,y,cols[cnt],label=labeltxt)
                    cnt+=1

            fig, ax = plt.subplots()

            plothist(self.trainloss_hist,'train_')
            plothist(self.validloss_hist,'')
            
            if self.model.saved_points is not None:
                for k in self.model.saved_points:
                    xx = int(self.model.saved_points[k])
                    plt.axvline(x=xx)
                    plt.text(int(xx), 14,k)
            
            ax.set_ylim(ymax=10)
            

#%%           
            plt.legend(loc=3)
            plt.grid()
            if self.model.modelname is not None:
                ps = os.path.split(self.model.modelname)
                tit = os.path.split(ps[0])[1] + "/" + ps[1] + " (" + str(self.model.train_cycle) + ")"
                plt.title(tit)
                plt.pause(0.001)  
                plt.savefig(self.model.modelname + ".png")
            else:
                plt.pause(0.001)  
#%%            
    except Exception as e:
        if 'DISPLAY' in os.environ:        
            print("Exception:" + str(e))
            print("problems during plotting. Wrong Display?? (DISPLAY="+os.environ['DISPLAY']+")")
        else:                                                                                       
            print("problems during plotting. No DISPLAY set!")
            
    #%%


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
               finalizeOnApply=False,

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
               saved_points = None,
               modelname = None,
               train_cycle = None,
               augment = None,
               align_physical = None,
               input_fdim = None
               ):
    super().__init__()
    self.preprocessor = []
    self.blocks = []
    self.classifiers = []
    self.finalBlock=finalBlock
    self.finalizeOnApply = finalizeOnApply
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
    self.train_cycle = train_cycle
    self.saved_points = saved_points
    self.augment = augment
    self.align_physical = align_physical
    self.input_fdim = input_fdim
    self.compiled = {}
    
    
    
    if modelname is not None and path.exists(modelname + ".json"):
         warnings.warn(modelname + ".json already exists!! Are you sure you want to override?")
        
    
    if self.block_out is None:
        self.block_out = []
        for k in range(self.cropper.depth-1): 
          self.block_out.append(num_labels+intermediate_out)
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
        
        
        
    for k in range(self.cropper.depth): 
      self.blocks.append(theBlockCreator(k))

    if preprocCreator is not None:
       for k in range(self.cropper.depth): 
           self.preprocessor.append(preprocCreator(level=k))
        
    if classifierCreator is not None:
       for k in range(self.cropper.depth): 
         clsfier = classifierCreator(level=k,outK=num_classes)
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
               'finalizeOnApply':self.finalizeOnApply,
               'trainloss_hist':self.myhist.trainloss_hist,
               'validloss_hist':self.myhist.validloss_hist,
               'trained_epochs':self.trained_epochs,
               'saved_points':self.saved_points,
               'train_cycle':self.train_cycle,
               'augment':self.augment,
               'align_physical':self.align_physical,
               'input_fdim':self.input_fdim
            }


  def set_cropgen(self, cropper):
      self.cropper = cropper
      cropper.model = self



  def call(self, inputs, training=False, lazyEval=None, stitch_immediate=False, testIT=False):

  
    nD = self.cropper.ndim
    
    ## squeeze additional batch_dim if necearray (max_agglomerative=True is true)
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
    output_nonspatial = []
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


      ## this is output from the last scale
      last = res

      ## get data and cropcoords at currnet scale
      if callable(inputs):
          inp,coords = inputs(idx=idx)          
      else:
          inp = inputs['input' + str(k)]
          coords = inputs['cropcoords' + str(k)]

      if k > 0: # it's not the initial scale, so do cropping etc.
          
         # in case tile results from last scale (for lazyeval or treemode)
         if last.shape[0] is not None and coords.shape[0] != last.shape[0]:         
            mtimes = coords.shape[0]//last.shape[0]
            multiples = [1]*(nD+2)
            multiples[0] = mtimes
            last = tf.tile(last,multiples)   
        
         # crop the relevant region
         if self.cropper.interp_type == 'lin':
             last_cropped = tf.gather_nd(last,tf.cast(0.5+coords,dtype=tf.int32),batch_dims=1)
         else:
             last_cropped = tf.gather_nd(last,coords,batch_dims=1)
         
         # if there is a preprocessor defined, apply it
         if k < len(self.preprocessor) :
             inp = self.preprocessor[k](inp,training=training)
         
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
             inp = self.preprocessor[k](inp,training=training)
          

      ## now, apply the network at the current scale       
      if testIT:
         res=inp
         output.append(res)
         
      else:
         res = self.blocks[k](inp,training=training)      
         if k < len(self.classifiers) and training:
            # res_nonspatial = self.classifiers[k](tf.concat([inp,res],nD+1),training=training) 
             res_nonspatial = self.classifiers[k](res,training=training) 
             output_nonspatial.append(res_nonspatial)
         
         outs = res[...,0:self.num_labels]
         
         ## apply a finalBlock on the last spatial output    
         if (training == False or not self.finalizeOnApply) and self.finalBlock is not None and k == self.cropper.depth-1:
               if isinstance(self.finalBlock,list):
                   for fb in self.finalBlock:
                       output.append(fb(outs))
               else:                    
                   output.append(self.finalBlock(outs,training=training))
         else:
             output.append(outs)
         

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
                
    if not self.intermediate_loss:
         output = [output[-1]]          
                
    if self.spatial_max_train:
        for k in range(len(output)):
            output[k] = tf.reduce_max(output[k],axis=list(range(1,nD+1)))
            
            
    if len(output_nonspatial) > 0:
        output_nonspatial = [tf.reduce_max(tf.concat(output_nonspatial,1),1)]
            
    return output + output_nonspatial


  def apply_full(self, data,
                 resolution=None,
                 level=-1,
                 generate_type='random',
                 jitter=0.05,
                 jitter_border_fix = False,
                 overlap=0,
                 repetitions=1,           
                 dphi=0,
                 augment=None,
                 branch_factor=1,
                 scale_to_original=True,
                 verbose=False,
                 init=False,
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


     if augment is None:
          augment = self.augment


              
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
                                    augment=augment,
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
            r = self(data_,lazyEval=lazyEval,stitch_immediate=stitch_immediate,testIT=testIT,training=init)
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
                r = r[0:self.cropper.depth]
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
                res[k] = tf.squeeze(resizeNDlinear(tf.expand_dims(res[k],0),orig_shape,True,nD,edge_center=True))                        
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
                 augment=None,
                 along4dim=False,
                 align_physical=None,
                 patch_size_factor=1,
                 crop_fdim=None,
                 crop_sdim=None,
                 testIT=False,
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
    
      if align_physical is None:
          align_physical = self.align_physical

      if align_physical is None:
          align_physical = True
              
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
                                augment=augment,
                                resolution = resolution,
                                lazyEval = lazyEval,
                                patch_size_factor=patch_size_factor,
                                verbose=True,
                                testIT=testIT,
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
          vsz = img1.header['pixdim'][1:nD+1]
          sz = a.shape[1:nD+1]
          if nD == 2:
              facs = [res.shape[0]/sz[0],res.shape[1]/sz[1],1]
          else:
              facs = [res.shape[0]/sz[0],res.shape[1]/sz[1],res.shape[2]/sz[2]]
          img1.header.set_data_shape(res.shape)
          newaffine = np.matmul(img1.affine,np.array([[1/facs[0],0,0,1],
                                                      [0,1/facs[1],0,1],
                                                      [0,0,1/facs[2],1],
                                                      [0,0,0,1]]))
            
      
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
     if self.train_cycle is not None:              
         self.saved_points[str(self.train_cycle)]=self.trained_epochs
     with open(outname,'w') as outfile:
         json.dump(self, outfile,cls=patchworkModelEncoder)
     print(fname +" saved!")
     if self.train_cycle is not None:         
         import os
         ps = list(os.path.split(fname))
         ps[-1] = ps[-1] + ".weights"
         outpath = "/".join(ps)
         if not os.path.exists(outpath):
             os.mkdir(outpath)
         ps.append("data."+str(self.train_cycle)+".tf")
         outname =  "/".join(ps)
     else:         
         outname = fname + ".tf"
     self.save_weights(outname,save_format='tf')

  @staticmethod
  def load(name,custom_objects={},show_history=False,immediate_init=True,notmpfile=False,train_cycle=-1,
           clsCreator=None     ):

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

    weightspath = name + ".tf"
    if train_cycle == -1:
        if model.train_cycle is not None:
            weightspath = name + ".weights/data."+str(model.train_cycle)+".tf"
    elif train_cycle is not None:
        weightspath = name + ".weights/data."+str(train_cycle)+".tf"
        
    print("loading weights from " + weightspath)

    notmpfile = True
    if immediate_init and model.input_fdim is not None: 

        if notmpfile:
            model.load_weights(weightspath)   
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
                        
            model.load_weights(loading_name_tmp + suffix)            
        
        
        try:
            if model.cropper.ndim == 3:
                initdat = tf.ones([1,32,32,32, model.input_fdim])
            else:
                initdat = tf.ones([1,32,32, model.input_fdim])    
            print("----------------- load/init network by minimal application")
            dummy = model.apply_full(initdat,resolution=[1,1,1],verbose=True,scale_to_original=False,generate_type='random',repetitions=1,init=False)        
            print("----------------- model and weights loaded")
        except:
            if not notmpfile:
                shutil.rmtree(tmpdir)
            raise 
        
        if not notmpfile:
            shutil.rmtree(tmpdir)
    else:        
        model.load_weights(weightspath)   


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
            train_ids = None,
            unlabeled_ids = [],
            valid_ids = [],
            valid_num_patches=None,
            self_validation=False,
            batch_size=32,
            verbose=1,
            steps_per_epoch=None,
            inc_train_cycle=True,
            jitter=0,
            jitter_border_fix=False,
            balance=None,
            showplot=True,
            autosave=True,
            augment=None,
            max_agglomerative=False,
            rot_intrinsic=0,
            loss=None,
            optimizer=None,
            patch_on_cpu=True,
            fit_type='keras',
            train_S = True,
            train_U = True,
            train_D = True,
            callback=None
            ):
      
    def f1_metric(y_true, y_pred):
        true_positives = kb.sum(kb.round(kb.clip(y_true * y_pred, 0, 1)))
        possible_positives = kb.sum(kb.round(kb.clip(y_true, 0, 1)))
        predicted_positives = kb.sum(kb.round(kb.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + kb.epsilon())
        recall = true_positives / (possible_positives + kb.epsilon())
        f1_val = 2*(precision*recall)/(precision+recall+kb.epsilon())
        return f1_val      
      

    def getSample(subset,np,valid=False):
        tset = [trainset[i] for i in subset]
        lset = [labelset[i] for i in subset]      
        rset = None
        
        dphi = rot_intrinsic
        aug_ = augment
        if valid:
            dphi=0
            aug_={'dphi':0,'dscale':0,'flip':0}
        
        self.cropper.dest_full_size = [None]*self.cropper.depth

        
        if resolutions is not None:
            rset = [resolutions[i] for i in subset]      
            
        with tf.device(DEVCPU):    
    
            if traintype == 'random':
                c = self.cropper.sample(tset,lset,resolutions=rset,generate_type='random', 
                                        num_patches=np,augment=aug_,balance=balance,dphi=dphi)
            elif traintype == 'tree':
                c = self.cropper.sample(tset,lset,resolutions=rset,generate_type='tree_full', jitter=jitter,
                                        jitter_border_fix=jitter_border_fix,augment=aug_,balance=balance,dphi=dphi)
        return c
    
    @tf.function
    def valid_step_supervised(images,lossfun,prefix):
            
      hist = {}
      
      data = images[0]
      labels = images[1]
      
     
      preds = self(data, training=False)
      loss = 0
      for k in range(len(labels)):          
            l = lossfun[k](labels[k],preds[k])
            l = tf.reduce_mean(l)
            loss += l
            if len(labels) > 1:
                if k == len(labels)-1:
                    hist[prefix+'_output_'+str(k+1)+'_loss'] = l
                    hist[prefix+'_output_'+str(k+1)+'_f1'] = 10**f1_metric(labels[k],preds[k])
  #    hist[prefix+'_S_loss'] = loss
      return hist
    
    def train_step_supervised(images,lossfun):
            
      hist = {}
      
      data = images[0]
      labels = images[1]
      
      trainvars = self.block_variables
              
      with tf.GradientTape() as tape:
        preds = self(data, training=True)
      
        loss = 0
        depth = len(labels)
        for k in range(depth):
            l = lossfun[k](labels[k],preds[k])
            l = tf.reduce_mean(l)
            if depth > 1:
                if k == depth-1:
                    hist['output_' + str(k+1) + '_loss'] = l
                    hist['output_' + str(k+1) + '_f1'] = 10**f1_metric(labels[k],preds[k])
            else:
                hist['output_loss'] = l
                hist['output_f1'] = 10**f1_metric(labels[k],preds[k])
                
            loss += l
            
      gradients = tape.gradient(loss,trainvars)
      self.optimizer.apply_gradients(zip(gradients, trainvars))
      return hist
  
    
    def train_step_discriminator(labeled,unlabeled):
      hist = {}

      trainvars = self.disc_variables
      
      # with tf.GradientTape() as tape:
      #   pred_label = self(labeled, training=True)
      #   pred_unlabel = self(unlabeled, training=True)
      #   loss =  tf.keras.losses.binary_crossentropy(pred_label[-1],tf.ones_like(pred_label[-1]))
      #   loss += tf.keras.losses.binary_crossentropy(pred_unlabel[-1],tf.zeros_like(pred_unlabel[-1]))
      #   hist['D_loss'] = loss
      # gradients = tape.gradient(loss,trainvars)
      # self.optimizer.apply_gradients(zip(gradients, trainvars))

      with tf.GradientTape() as tape:
        pred_label = self(labeled, training=True)
        loss1 =  tf.keras.losses.binary_crossentropy(tf.ones_like(pred_label[-1]),pred_label[-1])
      gradients = tape.gradient(loss1,trainvars)
      self.optimizer.apply_gradients(zip(gradients, trainvars))

      with tf.GradientTape() as tape:
        pred_unlabel = self(unlabeled, training=True)
        loss2 = tf.keras.losses.binary_crossentropy(tf.zeros_like(pred_unlabel[-1]),pred_unlabel[-1])
      gradients = tape.gradient(loss2,trainvars)
      self.optimizer.apply_gradients(zip(gradients, trainvars))

      hist['D_loss'] = loss1+loss2


      return hist
        
    def train_step_unsupervised(data,lam):
      hist = {}

      trainvars = self.block_variables
      
      with tf.GradientTape() as tape:
        pred = self(data, training=True)
        loss =  tf.keras.losses.binary_crossentropy(tf.ones_like(pred[-1]),pred[-1])
        hist['U_loss'] = loss
        loss = loss*lam

      gradients = tape.gradient(loss,trainvars)
      self.optimizer.apply_gradients(zip(gradients, trainvars))

      return hist
        
    # ---------------------------------------------------------------------------
    
    if patch_on_cpu:
        DEVCPU = "/cpu:0"
    else:
        DEVCPU = "/gpu:0"
    
    if self.saved_points is None:
        self.saved_points = {}
        
    if self.train_cycle is None:
       self.train_cycle = 0   
    if inc_train_cycle:
        self.train_cycle += 1
        
    self.block_variables = list([])
    for b in self.blocks:
        self.block_variables += b.trainable_variables
        
    self.disc_variables = list([])
    for b in self.classifiers:
        self.disc_variables += b.trainable_variables
    
    if isinstance(augment,dict):        
        self.augment = augment
    
    
    if valid_num_patches is None:
        valid_num_patches = num_patches
 
    if not hasattr(self,'optimizer') or self.optimizer is None:    
        if optimizer is None:
            optimizer = tf.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=True)
        self.optimizer = optimizer            

    if loss is None and not hasattr(self,'loss'):
        loss = []
        for k in range(self.cropper.depth-1):
            loss.append(lambda x,y: tf.keras.losses.binary_crossentropy(x,y,from_logits=True))
        loss.append(lambda x,y: tf.keras.losses.binary_crossentropy(x,y,from_logits=False))
        self.loss = loss
    else:
        self.loss = loss
    
    loss = self.loss
    
    
    if loss is not None and fit_type=='keras':
        print("compiling ...")
        self.compile(loss=loss, optimizer=self.optimizer)


    if train_ids is not None:
        trainidx = train_ids
    else:
        trainidx = list(range(len(trainset)))
    trainidx = [item for item in trainidx if item not in valid_ids]
    trainidx = [item for item in trainidx if item not in unlabeled_ids]

    GAN_train = len(unlabeled_ids) > 0
               
    print("labeled train imgs: "+ str(len(trainidx)))
    if GAN_train:
        print("unlabeled imgs: "+ str(len(unlabeled_ids)))        
    print("validate imgs: "+ str(len(valid_ids)))
            
    was_last_min = False
    for i in range(num_its):
        print("----------------------------------------- iteration:" + str(i))
        
        ### sampling
        print("sampling patches for training")
        start = timer()
        c_data = getSample(trainidx,num_patches)      
        end = timer()
        total_numpatches = len(trainidx)*num_patches
        if batch_size > total_numpatches:
            batch_size = total_numpatches        
        print("time elapsed, sampling: " + str(end - start) + " (for " + str(total_numpatches) + ")")
        if GAN_train:
            if total_numpatches == 0:
                unlabeled_num_patches = num_patches
            else:
                unlabeled_num_patches = total_numpatches // len(unlabeled_ids)
            start = timer()
            c_unlabeled_data = getSample(unlabeled_ids,unlabeled_num_patches)              
            end = timer()
            total_unlabeled_numpatches = len(unlabeled_ids)*unlabeled_num_patches
            if batch_size > total_unlabeled_numpatches:
                batch_size = total_unlabeled_numpatches
            print("time elapsed, sampling unlabeled data: " + str(end - start) + " (for " + str(total_unlabeled_numpatches) + ")")

        
        ### fitting
        sampletyp = [None, -1]
        if max_agglomerative:
            sampletyp[1] = num_patches

        debug=False
            

        print("starting training")
        start = timer()
        
        lam = tf.convert_to_tensor(0.1)


        if fit_type == 'custom':
            shuffle_buffer = max(2000,total_numpatches)
            dataset = c_data.getDataset().shuffle(shuffle_buffer).batch(batch_size,drop_remainder=True)

            if not "train_step" in self.compiled:
                if debug:
                    self.compiled["train_step"] = train_step_supervised
                else:
                    self.compiled["train_step"] = tf.function(train_step_supervised)
                
            
            numsamples = tf.data.experimental.cardinality(dataset).numpy() * batch_size
            infostr = '#samples: ' + str(numsamples) 

            unlabset = []
            numsamples_unl = 0
            if GAN_train:
                unlabdata = c_unlabeled_data.getInputData(sampletyp)
                unlabset = tf.data.Dataset.from_tensor_slices(unlabdata).shuffle(total_unlabeled_numpatches).batch(batch_size,drop_remainder=True)
                numsamples_unl = tf.data.experimental.cardinality(unlabset).numpy() * batch_size
                infostr += ', #unlabeled_samples: ' + str(numsamples_unl) 

            if not "train_step_discrim" in self.compiled:
                if debug:
                    self.compiled["train_step_discrim"] = (train_step_discriminator)
                    self.compiled["train_step_unsuper"] = (train_step_unsupervised)
                else:
                    self.compiled["train_step_discrim"] = tf.function(train_step_discriminator)
                    self.compiled["train_step_unsuper"] = tf.function(train_step_unsupervised)
            
            for e in range(epochs):
                print("EPOCH " + str(e+1) + "/"+str(epochs),end=',  ')
                print(infostr + ", batchsize: " + str(batch_size) , end=',  ')
                sttime = timer()
                log = []
                diter = iter(dataset)
                uiter = iter(unlabset)
                while True:
                    losslog = {}
                    labeled_element = next(diter,None)
                    if labeled_element is not None and train_S:
                        losslog.update( self.compiled["train_step"](labeled_element,loss) )
                    if GAN_train:    
                        unlabeled_element = next(uiter,None)
                        if unlabeled_element is not None:
                            if labeled_element is not None and train_D:                            
                                losslog.update( self.compiled["train_step_discrim"](labeled_element[0],unlabeled_element) )
                            if train_U:
                                losslog.update( self.compiled["train_step_unsuper"](unlabeled_element,lam) )
                    if labeled_element is None and (not GAN_train or unlabeled_element is None):
                        break
                    
                    log.append(losslog)
                    print('.', end='')                
                
                log, new_min = self.myhist.accum('train',log,1,tensors=True,mean=True)
                self.trained_epochs+=1
                end = timer()
                print( "  %.2f ms/sample " % (1000*(end-sttime)/(numsamples+numsamples_unl)))
                for k in log:
                    print(k + ":" + str(log[k][0]),end=" ")
                print("")

        else:
            inputdata = c_data.getInputData(sampletyp)
            targetdata = c_data.getTargetData(sampletyp)
            
            history = History()
            self.fit(inputdata,targetdata,
                      epochs=epochs,
                      verbose=verbose,
                      steps_per_epoch=steps_per_epoch,
                      batch_size=batch_size,
                      callbacks=[history])
            self.myhist.accum('train',history.history,epochs)
            self.trained_epochs += epochs
            del inputdata
            del targetdata
        end = timer()
        
        del c_data.scales
        del c_data

                    
        print("time elapsed, fitting: " + str(end - start) )
        
        
        

        ### validation
        def do_validation(idx,npatches,prefix):
            c = getSample(idx,npatches,valid=True)    
            print("validating")
            dataset = c.getDataset().batch(batch_size)
            log = []
            for images in dataset:
                losslog = valid_step_supervised(images,loss,prefix)            
                log.append(losslog)
                print('.', end='')                
            print('| ')                                
            log, new_min = self.myhist.accum(prefix,log,1,tensors=True,mean=True)
            for k in log:
                print(k + ":" + str(log[k][0]),end=" ")
            print("")
            return new_min

        
        if self_validation:        
            print("sampling patches for self validtion")
            new_min_self = do_validation(trainidx,max(5,num_patches//10),'selfv')
        
        if len(valid_ids) > 0:
            print("sampling patches for validtion")
            new_min_valid = do_validation(valid_ids,valid_num_patches,'valid')
            

            
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
        
