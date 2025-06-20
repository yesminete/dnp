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
import os
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
        self.handles = [sys.__stderr__]
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

  def getNumberOfSeenPatches(self):
      if self.trainloss_hist is None:
          return 0
      k = list(self.trainloss_hist.keys())
      if len(k) > 0:
          return self.trainloss_hist[k[0]][-1][0]          
      else:
          return 0

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
         #    loss_hist[k] += list(zip(list(range(self.model.trained_epochs,self.model.trained_epochs+epochs)),cur_hist[k]))
             bsize = epochs//len(cur_hist[k])
             loss_hist[k] += list(zip(list(range(self.model.trained_epochs,self.model.trained_epochs+epochs,bsize)),cur_hist[k]))
             
     
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

#    try: 
        
        import warnings
        warnings.filterwarnings('ignore', '.*plotlib is currently using agg.*', )

        import matplotlib as mpl
    #    mpl.use('Agg')

        import matplotlib.pyplot as plt
        from matplotlib import gridspec
        
        plt.rcParams.update({'font.size': 9})
        

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
            
            def plothist(loss_hist,txt,nodisp=False):
                import matplotlib.pyplot as plt
                
                cols = 'rbmckyrbmckyrbmcky'   
                cnt = 0
                for k in sorted(loss_hist):
                    if nodisp:
                      if not k.find('nodisplay') > -1:
                        continue
                    else:
                      if k.find('nodisplay') > -1:
                        continue
                    x = [ i/1000 for i, j in loss_hist[k] ]
                    y = [ j for i, j in loss_hist[k] ]
                    x_orig = x
                    y_orig = y
                    n = int(np.ceil(len(y)/20.0))
                    y = np.convolve(y,np.ones([n])/n,mode='valid')
                    if n > 1:
                        y = np.concatenate([np.ones([n-1])*y[0],y],0)
                    
                    labeltxt = (txt+k).replace("_output","") + (": {:.2f}").format(math.log(y[-1])/math.log(10.0)   )
                    
                    
                    if txt == "":                     
                        if labeltxt.find("_threshold")>-1:
                            plt.semilogy(x,y,'y',label=labeltxt,linestyle='dotted')
                        else:
                            plt.semilogy(x,y,cols[cnt],label=labeltxt,marker='o', linestyle='dashed')
                    else:
                        if labeltxt.find("_threshold")>-1:
                            plt.semilogy(x,y,'y',label=labeltxt, linestyle='dashdot')
                        else:    
                            plt.semilogy(x,y,cols[cnt],label=labeltxt)
                            plt.semilogy(x_orig,y_orig,cols[cnt],label=labeltxt,alpha=.3)
                    cnt+=1

 
            
            fig = plt.figure(1,figsize=(7, 10))

            #_, ax = plt.subplots()
            if hasattr(self,'age'):
                gs = fig.add_gridspec(3, 2, width_ratios=[3, 1],height_ratios=[3,1,1]) 
                ax_age = 1
                ax_bal = 2
                ax_f1 = 4
                ax_f1add = 5
            else:
                gs = fig.add_gridspec(3, 1, height_ratios=[3,1,1]) 
                ax_bal = 1
                ax_f1 = 2
                ax_f1add = 3


            ax = plt.subplot(gs[0])
            plothist(self.trainloss_hist,'train_')
            plothist(self.validloss_hist,'')

            
            if self.model.saved_points is not None:
                for k in self.model.saved_points:
                    xx = int(self.model.saved_points[k]/1000)
                    plt.axvline(x=xx)
                    plt.text(int(xx), 14,k)
            
            ax.set_ylim(top=10)
            if self.model.modelname is not None:
                ps = os.path.split(self.model.modelname)
                tit = os.path.split(ps[0])[1] + "/" + ps[1] + " (" + str(self.model.train_cycle) + ")"
                plt.title(tit)
            

#%%           

            handles_, labels_ = ax.get_legend_handles_labels()
            labels_unique = {}
            labels = []
            handles = []
            for k in range(len(labels_)):
                if labels_[k] not in labels_unique:
                    labels_unique[labels_[k]] = 1
                    labels.append(labels_[k])
                    handles.append(handles_[k])
        
            ax.legend(handles, labels,loc=3,fontsize=7)

            plt.grid()
                
            
            if hasattr(self,'age'):
                ax1 = plt.subplot(gs[ax_age])
                plt.barh(range(self.age.shape[0]),tf.sort(self.age).numpy())
                plt.yticks([])
                plt.title('hard patch age')
                plt.grid()

        
            ax2 = plt.subplot(gs[ax_bal])            
            if hasattr(self,"pixelratio"):
                b = tf.squeeze(self.pixelratio[-1])
                if len(b.shape) > 0:
                    plt.bar(range(len(b)),b)
                    plt.title('balances')            
                    ax2.set_ylim([0, 1])
                    plt.xticks(range(len(b)))
                    if hasattr(self,'categorial_label') and self.categorial_label is not None:
                        ax2.set_xticklabels(self.categorial_label)
                    else:
                        ax2.set_xticklabels(range(1,1+len(b)))
        
                    ax2 = plt.subplot(gs[ax_f1])
                    if 'valid_nodisplay_class_f1' in self.validloss_hist:
                        b = tf.squeeze(self.validloss_hist['valid_nodisplay_class_f1'][-1][1])
                        plt.title('f1 scores (valid)')            
                    elif 'nodisplay_class_f1' in self.trainloss_hist:
                        b = tf.squeeze(self.trainloss_hist['nodisplay_class_f1'][-1][1])
                        plt.title('f1 scores (train)')            
                    if len(b.shape) > 0:
                        plt.bar(range(len(b)),b)
                        ax2.set_ylim([0, 1])
                        plt.xticks(range(len(b)))
                        if hasattr(self,'categorial_label') and self.categorial_label is not None:
                            ax2.set_xticklabels(self.categorial_label)
                        else:
                            ax2.set_xticklabels(range(1,1+len(b)))
    

     #       ax2 = plt.subplot(gs[ax_f1add])            
     #       plothist(self.trainloss_hist,'train_',True)



            if self.model.modelname is not None:
                plt.pause(0.001)  
                plt.show()
                fig.savefig(self.model.modelname + ".png")                
            #    plt.savefig(self.model.modelname + ".png")
            else:
                plt.pause(0.001)  


                 
#%%            
    # except Exception as e:
    #     if 'DISPLAY' in os.environ:        
    #         print("Exception:" + str(e))
    #         print("problems during plotting. Wrong Display?? (DISPLAY="+os.environ['DISPLAY']+")")
    #     else:                                                                                       
    #         print("problems during plotting. No DISPLAY set!")
            
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
               space_loss = None,
               finalBlock=None,
               finalizeOnApply=True,
               finalBlock_all_levels=False,

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
               info = {},
               crop_fdim = None,
               input_fdim = None
               ):
    super().__init__()
    self.preprocessor = []
    self.blocks = []
    self.classifiers = []
    self.finalBlock=finalBlock
    self.finalBlock_all_levels = finalBlock_all_levels
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
    self.space_loss = space_loss
    if self.space_loss is not None:
        self.finalizeOnApply = True
    
    
    
    self.trained_epochs = trained_epochs
    self.modelname = modelname
    self.train_cycle = train_cycle
    self.saved_points = saved_points
    self.augment = augment
    self.info = info
    self.align_physical = align_physical
    self.crop_fdim = crop_fdim
    self.input_fdim = input_fdim
    self.compiled = {}
    
    
    
    if modelname is not None and path.exists(modelname + ".json"):
         warnings.warn(modelname + ".json already exists!! Are you sure you want to override?")
        
    
    if self.block_out is None:
        self.block_out = []
        
        if cropper.categorial_label_original is not None:
            if cropper.categorical:
                num_labels = len(cropper.categorial_label_original)+1
            else:
                num_labels = len(cropper.categorial_label_original)
                
        if num_labels != -1:
            for k in range(self.cropper.depth-1): 
              self.block_out.append(num_labels+intermediate_out)
            self.block_out.append(num_labels)
        else:
            for k in range(self.cropper.depth): 
              self.block_out.append(intermediate_out)
    

    import inspect
    signature = inspect.signature(blockCreator)

    if 'input_shape' in signature.parameters and 'inK' in signature.parameters:
        assert self.input_fdim is not None, "pass input_fdim to the model when using outK in the blockCreator"
            
        def blkCreator_(level):
            if level == 0:
                return blockCreator(level=level,outK=self.block_out[level],input_shape=cropper.get_patchsize(level),inK=self.input_fdim) 
            else:
                return blockCreator(level=level,outK=self.block_out[level],input_shape=cropper.get_patchsize(level),
                                    inK=self.input_fdim+self.block_out[level-1]) 
        blkCreator = blkCreator_
    elif 'input_shape' in signature.parameters:
        blkCreator = lambda level:  blockCreator(level=level,outK=self.block_out[level],input_shape=cropper.get_patchsize(level))   
    else:
        blkCreator = lambda level:  blockCreator(level=level,outK=self.block_out[level])
        
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
    
    self.external_trainvars = []
    if self.space_loss is not None:
        self.space_offset = tf.Variable(tf.zeros([self.cropper.depth]))
        self.external_trainvars += [self.space_offset]

  
  def serialize_(self):
    return   { 'forward_type':self.forward_type,

               'blocks':self.blocks,
               'intermediate_out':self.intermediate_out,
               'intermediate_loss':self.intermediate_loss,   
               'identical_blocks':self.identical_blocks,   
               'block_out':self.block_out,   
               'spatial_train':self.spatial_train,
               'space_loss':self.space_loss,
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
               'finalBlock_all_levels':self.finalBlock_all_levels,
               'trainloss_hist':self.myhist.trainloss_hist,
               'validloss_hist':self.myhist.validloss_hist,
               'trained_epochs':self.trained_epochs,
               'saved_points':self.saved_points,
               'train_cycle':self.train_cycle,
               'augment':self.augment,
               'info':self.info,
               'align_physical':self.align_physical,
               'crop_fdim':self.crop_fdim,
               'input_fdim':self.input_fdim
            }


  def set_cropgen(self, cropper):
      self.cropper = cropper
      cropper.model = self



  def call(self, inputs, training=False, lazyEval=None, stitch_immediate=False, batch_size=None, testIT=False):


    def batcher(x,fun):
        if batch_size is not None:          
            d = x.shape[0]
            n = d//batch_size + 1
            r = []
            for i in range(n):
                r.append(fun(x[i*batch_size:min((i+1)*batch_size,d)]))
            return tf.concat(r,0)
        else:
            return fun(x)
  
    nD = self.cropper.ndim
    
    ## squeeze additional batch_dim if necearray (max_agglomerative=True is true)
    original_shape = None
    if not callable(inputs) and len(inputs['input0'].shape) > nD+2:
        original_shape = inputs['input0'].shape[1]
        for k in inputs:
            if k != "dest_full_size" and k != 'parent_box_scatter_index' and k != 'batchindex':
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
              
      if not callable(inputs):
          if ('input'+str(k)) not in inputs:
             break

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
                if hasattr(res,'QMembedding'):
                    attention = reduceFun(attentionFun(res[...,0:res.QMembedding.embedD]),axis=list(range(1,nD+2)))                  
                else:
                    if self.cropper.categorical:
                        attention = reduceFun(attentionFun(res[...,1:self.num_labels]),axis=list(range(1,nD+2)))                  
                    else:
                        attention = reduceFun(attentionFun(res[...,0:self.num_labels]),axis=list(range(1,nD+2)))                  
            else:
                attention = reduceFun(attentionFun(res[...,lazyEval['label']:lazyEval['label']+1]),axis=list(range(1,nD+2)))                  
                
        idx = tf.argsort(attention,0,'DESCENDING')
        if fraction < 1:
            numps = tf.cast(tf.floor(idx.shape[0]*fraction),dtype=tf.int32)
            if numps == 0:
                numps = 1
            idx = idx[0:numps]            

        if 'batchselection' in lazyEval:
            idx = lazyEval['batchselection'](idx,k)

        #print('lazyEval, level ' + str(k-1) + ': only forwarding ' + str(idx.shape[0]) + ' patches to next level')
        res = tf.gather(res,idx,axis=0)
        if res_nonspatial is not None:
           res_nonspatial = tf.gather(res_nonspatial,idx,axis=0)


      ## this is output from the last scale
      last = res

      ## get data and cropcoords at currnet scale
      if callable(inputs):
          inp,coords = inputs(idx=idx,output_cropped=last)          
      else:
          inp = inputs['input' + str(k)]
          coords = inputs['cropcoords' + str(k)]

    #  if lazyEval is not None:
      #print('processing level ' + str(k) + ': ' + str(inp.shape[0]) + ' patches')


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
         if self.forward_type == 'noinput':
             inp = last_cropped
         elif self.forward_type == 'simple' or  self.forward_type == 'bridge' :
             if testIT == True:
                inp = last_cropped
             else:
                inp = tf.concat([inp,last_cropped],(nD+1))
         elif self.forward_type == 'mult':
            x = tf.expand_dims(inp,nD+2) * tf.sigmoid(tf.expand_dims(last_cropped,nD+1))
            x = tf.reshape(x,x.shape[0:nD+1] + x.shape[nD+1]*x.shape[nD+2])
            inp = tf.concat([inp,last_cropped,x],(nD+1))
         else: 
            assert 0,"unknown forward type"

      else:
         if k < len(self.preprocessor) :
             inp = self.preprocessor[k](inp,training=training)
          

      ## now, apply the network at the current scale       
      if testIT==True:
         res=inp
         output.append(res)
         
      else:
                    
         res = batcher(inp,lambda x: self.blocks[k](x,training=training))

         if self.forward_type == 'bridge' and k>0:
             res = res + last_cropped[...,0:res.shape[-1]]


         if k < len(self.classifiers) and self.classifier_train:
             res_nonspatial = self.classifiers[k](tf.concat([inp,res],nD+1),training=training) 
             output_nonspatial.append(res_nonspatial)
         
         def applyFB(fb,res):
           fbres = fb(res)
           if hasattr(res,'deep_out'):
              fbres.deep_out = res.deep_out
           return fbres
                      
         ## apply a finalBlock on the last spatial output    
         if self.spatial_train:
             hasblock = self.finalBlock is not None
             if (training == False or not self.finalizeOnApply) and (self.finalBlock is not None) and (k == self.cropper.depth-1 or self.finalBlock_all_levels):
                   if isinstance(self.finalBlock,list):
                       for fb in self.finalBlock:
                           output.append(applyFB(fb,res))
                   else:      
                       output.append(applyFB(lambda x: self.finalBlock(x,training=training),res))
             else:
                 output.append(res)
         

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
                #output[k] = tf.reduce_max(output[k],axis=1)
                
            if len(output_nonspatial) > 0:
                tmp = tf.concat(list(map(lambda x: tf.expand_dims(x,2),output_nonspatial)),2)
                tmp = tf.reshape(tmp,[tmp.shape[0]//original_shape, original_shape] + tmp.shape[1:])                
                tmp = tf.reduce_max(tmp,[1,3])
                output_nonspatial = [tmp]
                
                
    if not self.intermediate_loss and len(output)>0:
         output = [output[-1]]          
                
    if self.spatial_max_train:
        for k in range(len(output)):
            output[k] = tf.reduce_max(output[k],axis=list(range(1,nD+1)))
            

    #if len(output_nonspatial) > 0:
    #    output_nonspatial = [tf.reduce_max(tf.concat(list(map(lambda x: tf.expand_dims(x,2),output_nonspatial)),2),2)]
            
            
    if len(output_nonspatial) > 0 and not training:
        output_nonspatial = [tf.reduce_max(tf.concat(output_nonspatial,1),1,keepdims=True)]
        return output_nonspatial
            
    
    
    
    return output_nonspatial + output


  def apply_full(self, data,
                 resolution=None,
                 level=-1,
                 generate_type='random',
                 snapper=None,
                 jitter=0.05,
                 jitter_border_fix = False,
                 overlap=0,
                 repetitions=1,         
                 num_patches=None,
                 dphi=0,
                 augment=None,
                 branch_factor=None,
                 scale_to_original=True,
                 verbose=False,
                 init=False,
                 num_chunks=1,
                 patch_size_factor=1,
                 sampling_factor=1,
                 lazyEval = None,
                 window = None,
                 sparse_suppression=None,
                 max_patching=False,
                 patch_stats= False,
                 stitch_immediate=False,
                 QMapply_paras={},
                 testIT=False,
                 votemap=False
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
     
     pred = [0.0] * self.cropper.depth
     sumpred = [0.0]  * self.cropper.depth
     
     
     if num_patches is not None:
         repetitions = num_patches
     
     
     reps = 1
     if generate_type == 'random' or generate_type == 'random_fillholes' or generate_type == 'random_deprec' :
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
             if 'batches' in lazyEval:
                 lazyEval['fraction'] = 1
             else:
                 lazyEval['fraction'] = 1
         if branch_factor is None and generate_type != 'random':
             branch_factor = round(1/lazyEval['fraction'])
             print("branch factor is " + str(branch_factor))
         if 'label' not in lazyEval:
             lazyEval['label'] = None
         if 'batches' in lazyEval:
             if lazyEval['batches'] is not None:
                 B = lazyEval['batches']
                 repetitions = B**self.cropper.depth
                 def batchselection(idx,level,instance):                    
                    s = (instance//(B**level))%B
                    n = idx.shape[0]
                    idx = idx[s*(n//B+1):(s+1)*(n//B+1)]
                    
                        
                    return idx

     if branch_factor is None:
         branch_factor = 1

     if augment is None:
          augment = self.augment

     mix_levels = False
              
     pstats = [None]*len(level)

     for w in range(num_chunks):
         if w > 0:
             print('------------------------------------------------ chunk: ' + str(w) + "/" +  str(num_chunks))
         for i in range(repetitions):
             
            if lazyEval is None:             
                print(">>> sampling patches for testing")
                start = timer()
            else:
                if 'batches' in lazyEval:
                   if lazyEval['batches'] is not None:
                       lazyEval['batchselection'] = lambda x,y : batchselection(x,y,i)

            label_dummy = None
            balance = None
            if generate_type == 'random_fillholes' and w > 0:
                covering = tf.expand_dims(tf.reduce_max(sumpred[-1],axis=-1,keepdims=True),0)
                label_dummy = tf.cast(covering==0,dtype=tf.float32)
                balance = {"ratio":0.9}
                #print(data.shape)                
                #print(label_dummy.shape)
            if lazyEval is not None and 'selfref' in lazyEval:
                balance = {"selfref":lazyEval['selfref']}

            x = self.cropper.sample(data,label_dummy,test=False,generate_type=generate_type,snapper=snapper,
                                    resolutions=resolution,
                                    jitter = jitter,
                                    jitter_border_fix = jitter_border_fix,
                                    overlap=overlap,
                                    num_patches=reps,
                                    balance=balance,
                                    dphi=dphi,
                                    augment=augment,
                                    branch_factor=branch_factor,
                                    patch_size_factor=patch_size_factor,
                                    destshape_size_factor=sampling_factor,
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

            print(">>> applying network --- " + str(i+1) + "/" + str(repetitions))
            start = timer()

            if generate_type=='tree':
                r = self(data_,lazyEval=lazyEval,stitch_immediate=stitch_immediate,testIT=testIT,training=False,batch_size=32)
            else:
                r = self(data_,lazyEval=lazyEval,stitch_immediate=stitch_immediate,testIT=testIT,training=False)

                
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
                    stat = {'max': tf.reduce_max(r[k]), #,axis=list(range(1,len(sz)-1))),
                                   'mean': tf.reduce_mean(r[k]),#,axis=list(range(1,len(sz)-1))),
                                   'mean2': tf.reduce_mean(tf.math.pow(r[k],2)) #,axis=list(range(1,len(sz)-1))),
                                   }
                    if pstats[k] is None:
                        pstats[k] = stat
                    else:
                        for t in pstats[k]:
                            pstats[k][t] = tf.concat([pstats[k][t],stat[t]],0)
                     
            if not stitch_immediate>1:
                print(">>> stitching result")
                with tf.device("/cpu:0"):                
                    
                    start = timer()
                    r = r[0:self.cropper.depth]
                    if (self.spatial_train or max_patching) and not self.spatial_max_train:
                        if isinstance(level[0],str) and level[0].find('mix') != -1:
                            s = 0
                            if level[0].find('nohead') == -1 and len(level[0])>3:
                                s=int(level[0][3:])
                            level_to_stitch = list(range(s,self.cropper.depth))
                            print('mixing: ' + str(level_to_stitch))
                            mix_levels = True
                            if level[0].find('nohead') == -1 and not hasattr(r[0],'QMembedding'):
                                for k in level_to_stitch:            
                                  if k < self.cropper.depth-1:
                                    if self.cropper.categorical:
                                        r[k] = tf.nn.softmax(r[k])        
                                    else:
                                        r[k] = tf.nn.sigmoid(r[k])        
                        else:
                            level_to_stitch = level
                        for k in level_to_stitch:            
                          a,b = x.stitchResult(r,k,window=window)                                                       
                          pred[k] += a
                          sumpred[k] += b         
                        
                  
                        print(">>> #patches last level: " + str(r[-1].shape[0]))
                        print(">>> coverage: " + str(round(100*(tf.reduce_sum(tf.cast(sumpred[level_to_stitch[-1]][...,0]>0,dtype=tf.float32))/ tf.cast(tf.reduce_prod(sumpred[level_to_stitch[-1]].shape[0:nD]),dtype=tf.float32)).numpy())) + "%")
                        print(">>> time elapsed, stitching: " + str(timer() - start) )
                      

              
     if (self.spatial_train  or max_patching)  and not self.spatial_max_train:
         if stitch_immediate>1:
             res = r
             res[0] = tf.reduce_mean(res[0],axis=0)
         else:
            if testIT == 2:
                res = zipper(pred,sumpred,lambda a,b : (b+0.00001) )     
            else:
                if mix_levels:
                    pred_votes = 0
                    pred_on = 0
                    dest_shape = pred[-1].shape
                    print(">>> mixing ")    
                    start = timer()        
                    last_fdim = pred[-1].shape[-1]
                    for k in range(len(pred)):
                        if isinstance(pred[k],float):
                            continue;
                        fac = np.math.pow(0.2,len(pred)-1-k)
                        pr = tf.squeeze(resizeNDlinear(pred[k][...,0:last_fdim],dest_shape,nD=nD))
                        vr = tf.squeeze(resizeNDlinear(sumpred[k][...,0:last_fdim],dest_shape,nD=nD))
                        if sparse_suppression is not None:
                            vr = tf.math.sqrt(vr**2+sparse_suppression*3)
                        if len(pr.shape) == len(vr.shape)+1:
                            vr = tf.expand_dims(vr,-1)
                        pred_on = pred_on + fac*tf.cast(vr>0,dtype=tf.float32)
                        pred_votes = pred_votes + fac*tf.squeeze(pr/(vr+0.00001))
                    res = [pred_votes / (pred_on+0.00001)]
                    if len(res[0].shape) == nD:
                        res[0] = tf.expand_dims(res[0],-1)
                    level = [0]                    
                    print(">>> time elapsed, mixing: " + str(timer() - start) )

                else:
                    if sparse_suppression is not None:
                        res = zipper(pred,sumpred,lambda a,b : a/tf.math.sqrt(b**2+sparse_suppression*3) )    
                    else:
                        res = zipper(pred,sumpred,lambda a,b : a/(b+0.00001) )    
                          
                    if votemap:
                        for k in level:                        
                            res[k] = tf.concat([res[k],sumpred[k]],-1)
                        
             
                
         if isinstance(sampling_factor,list) or sampling_factor > 1: 
             if not isinstance(sampling_factor,list):
                 sigma = sampling_factor-1
             else:
                 if sampling_factor[-1] == 'mm':
                     dvsz = tf.cast(sampling_factor[0:-1],dtype=tf.float32)
                     sigma = resolution['voxsize']/dvsz-1
                 else:
                     sigma = sampling_factor-1
             if all(sigma > 0):
                 for k in level:
                    if nD==2:
                       res[k] = tf.squeeze(conv_gauss2D_fft(tf.expand_dims(res[k],0),sigma),0)
                    if nD==3:
                       res[k] = tf.squeeze(conv_gauss3D_fft(tf.expand_dims(res[k],0),sigma),0)
            
           
         sz = data.shape
         orig_shape = sz[1:(nD+1)]
         if scale_to_original:
             with tf.device("/cpu:0"):                
                 for k in level:
                    res[k] = tf.squeeze(resizeNDlinear(res[k],orig_shape,nD=nD))
         
        
         maxidxperlabel = None
            
         if hasattr(r[0],'QMembedding') and not init:
            print("doing QM prediction")
            for k in level:
               if votemap:
                   tmp = r[0].QMembedding.apply(res[k][...,:-1],params=QMapply_paras)     
                   if isinstance(tmp,tuple):
                       maxidxperlabel = tmp[1]                   
                       tmp = tmp[0]
                   res[k] = tf.concat([tmp,tf.cast(res[k][...,-1:],dtype=tf.int32)],-1)
               else:
                   tmp = r[0].QMembedding.apply(res[k],params=QMapply_paras)
                   if isinstance(tmp,tuple):
                       res[k] = tmp[0]
                       maxidxperlabel = tmp[1]
                   else:
                       res[k] = tmp
                   

         if single:
           res = res[level[0]]

        
         end = timer()
         print(">>> total time elapsed: " + str(end - start_total) )
         
         if patch_stats:
             if single:
                 pstats = pstats[level[0]]
             return res,pstats
         else:
             if maxidxperlabel is not None:
                 return res,maxidxperlabel
             else:
                 return res
     
     return r[0]

  # for multi-contrast data fname is a list
  def apply_on_nifti(self,fname, ofname=None,
                 generate_type='tree',
                 snapper=None,
                 overlap=0,
                 jitter=0.05,
                 jitter_border_fix=False,
                 repetitions=5, # deprecated
                 num_patches=None,
                 branch_factor=None,
                 num_chunks=1,
                 scale_to_original=True,
                 scalevalue=None,
                 dphi=0,
                 level=-1,
                 augment=None,
                 along4dim=False,
                 squeeze=False,
                 align_physical=None,
                 ce_threshold=0.01,
                 window=None,
                 sparse_suppression=None,
                 patch_size_factor=1,
                 sampling_factor=1,                 
                 crop_fdim=None,
                 crop_sdim=None,
                 out_typ='int16',
                 postproc=None,
                 ccana=None,
                 label_names=None,
                 label_colors=None,
                 testIT=False,
                 votemap=False,
                 verbose=False,
                 return_nibabel=True,
                 QMapply_paras={},
                 deprec_2d_res=False,
                 lazyEval = None):
        

      def conncomp(data,idx=None):
            if ccana is None:
                return data

            import cc3d    
            print('computing connected components')
            f = getattr(cc3d,ccana['typ'])    
            if 'params' in ccana:
                params = ccana['params']
            else:
                if ccana['typ'] == 'largest_k':
                    params = {'k':1}
                elif ccana['typ'] == 'dust':
                    params = {'threshold':100}
            g = lambda x: f(x,**params)
            if nD == 3:
                if len(data.shape) == 3:
                    data = g(data)
                else:
                   if idx is not None:
                       data_old = data[:,:,:,0]
                       data = data_old*0
                       for j in idx:
                          c = g(np.where(data_old==j,1.0,0.0))
                          data = np.where(c,j,data)
                       data = tf.expand_dims(data,-1)
                   else:
                       for k in range(0,data.shape[3]):
                          print('.')
                          data[:,:,:,k] = g(data[:,:,:,k])
            if nD == 2:
                if len(data.shape) == 2:
                    data = g(data)
                else:
                   if idx is not None:
                       data_old = data[:,:,0]
                       data = data_old*0
                       for j in idx:
                          c = g(np.where(data_old==j,1.0,0.0))
                          data = np.where(c,j,data)
                       data = tf.expand_dims(data,-1)
                   else:
                        for k in range(0,data.shape[2]):
                           print('.')
                           data[:,:,k] = g(data[:,:,k])
            return data

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
                if c[0] is not None:
                    img = img[c[0],...]
                if c[1] is not None:
                    img = img[:,c[1],...]
                if c[2] is not None:
                    img = img[:,:,c[2],...]
            return img,c
        else:
            return img,None
    
    
      if align_physical is None:
          align_physical = self.align_physical
      if align_physical is None:
          align_physical = False
              
      if crop_fdim is None:
          crop_fdim = self.crop_fdim
      if align_physical is None:
          crop_fdim = False


      if label_names is None:
          if self.info is not None:              
              if 'annotations_selector' in self.info:
                  if 'labels' in self.info['annotations_selector']:
                    lbls = self.info['annotations_selector']['labels']
                    label_names = []
                    for l in lbls:
                        label_names.append(l[0].split(".")[1])
                          
                  
          

      if hasattr(self.finalBlock,'isQMembedding'):
          out_typ = 'idx'
        
      scrop = None
     
      nD = self.cropper.ndim
      if not isinstance(fname,list):
          fname = [fname]
      ims = []


      template_nii = None      
      for f in fname:          
          img1 = nib.load(f)        
        
          if template_nii is not None:
              sz1 = img1.header.get_data_shape()
              sz2 = template_nii.header.get_data_shape()
              if np.abs(sz1[0]-sz2[0]) > 0 or np.abs(sz1[1]-sz2[1]) > 0 or np.abs(sz1[2]-sz2[2]) > 0 or np.sum(np.abs(template_nii.affine-img1.affine)) > 0.01:                           
                  img1 = resample_from_to(img1, template_nii,order=3)

          
          if template_nii is None:
              template_nii = img1
          
              if img1.shape[2] == 1:
                  if deprec_2d_res:
                      resolution = np.sqrt(np.sum(img1.affine[:,0:2]**2,axis=0))
                  else:
                      resolution = {"voxsize": img1.header['pixdim'][1:3], "input_edges":img1.affine}                  
              else:
                  if align_physical:
                      img1 = align_to_physical_coords(img1)
                      resolution = img1.header['pixdim'][1:4]
                  else:
                      resolution = {"voxsize":img1.header['pixdim'][1:4],"input_edges":img1.affine}




          a = img1.get_fdata()
          
          if crop_fdim is not None and crop_fdim:
             if len(a.shape) > nD:
                if crop_fdim == 'mean':
                    a = np.mean(a,axis=-1)
                else:                    
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
                                        snapper=snapper,
                                        jitter=jitter,
                                        jitter_border_fix=jitter_border_fix,
                                        overlap=overlap,                            
                                        repetitions=repetitions,
                                        num_patches=num_patches,
                                        num_chunks=num_chunks,
                                        branch_factor=branch_factor,
                                        dphi=dphi,
                                        augment=augment,
                                        resolution = resolution,
                                        level = level,
                                        lazyEval = lazyEval,
                                        patch_size_factor=patch_size_factor,
                                        sampling_factor=sampling_factor,
                                        window=window,
                                        sparse_suppression=sparse_suppression,
                                        verbose=verbose,
                                        testIT=testIT,
                                        votemap=votemap,
                                        QMapply_paras=QMapply_paras,
                                        scale_to_original=scale_to_original)

      if along4dim:
          res = []          
          for k in range(a.shape[nD+1]):
              res.append(tf.expand_dims(do_app(a[...,k:k+1]),nD))
          res = tf.concat(res,nD)             
          if along4dim == 'mean':
              res = tf.reduce_mean(res,axis=-1,keepdims=True)
          
      else:         
          res = do_app(a)
          
      maxidxperlabel = None
      if isinstance(res,tuple):
          maxidxperlabel = res[1]
          res = res[0]
          
      if postproc is not None:
         res = postproc(res)
          
      if hasattr(res,'numpy'):
          res = res.numpy()
          
      if nD == 2:
          if len(res.shape) == 3:         
              res = np.reshape(res,[res.shape[0],res.shape[1],1,res.shape[2]])
          if along4dim == True:
              res = np.reshape(res,[res.shape[0],res.shape[1],1,res.shape[2],res.shape[3]])
          img1.header.set_data_shape(res.shape)
          
      maxi = tf.reduce_max(tf.abs(res))
      
      newaffine = img1.affine
      if scrop is not None:
          offset = np.matmul(newaffine,(np.array([scrop[0][0],scrop[1][0],scrop[2][0],1])))
          newaffine[:,3] = offset
          
      
      if not scale_to_original:
          vsz = img1.header['pixdim'][1:nD+1]
          sz = a.shape[1:nD+1]
          facs = [1]*nD
          offs = [0]*nD
          for k in range(nD):
              facs[k] = (res.shape[k]-1)/(sz[k]-1)
              offs[k] = vsz[k]*(1-facs[k])*0
          if nD ==2:
             facs.append(1)              
             offs.append(0)
          img1.header.set_data_shape(res.shape)
          newaffine = np.matmul(img1.affine,np.array([[1/facs[0],0,0,offs[0]],
                                                      [0,1/facs[1],0,offs[1]],
                                                      [0,0,1/facs[2],offs[2]],
                                                      [0,0,0,1]]))
      
      print('saving nifti')
      pred_nii = None
      if ofname is not None:
          def savenii(name,res_,out_typ,labelidx=None):       
             nonlocal res
             threshold = None
             if out_typ == 'idx':
                 fac = 1
                 threshold = 0.5
             elif out_typ == 'int16':
                 fac = 32000/maxi                    
             elif out_typ == 'uint8':
                 fac = 255/maxi                    
             elif out_typ == 'float32' or out_typ == 'float' or out_typ == 'single':
                 fac = 1        
             elif (out_typ.find('mask') != -1) | (out_typ.find('atls') != -1):

                 fac = 1
                 if len(out_typ) > 4:
                     ths = list(map(float,out_typ[5:].split(",")))
                     if len(ths) == 1:
                         threshold = ths[0]*np.ones([self.num_labels])
                     else:
                         threshold = np.array(ths)
                 else:
                     threshold = 0.5*np.ones([self.num_labels])
                     if 'valid_nodisplay_class_threshold' in self.myhist.validloss_hist :
                         th = self.myhist.validloss_hist['valid_nodisplay_class_threshold']
                         threshold = th[-1][1]
                 nD = self.cropper.ndim
                 if nD == 3 and self.num_labels > 1:
                     threshold = np.expand_dims(threshold,[0,1,2])
                 elif nD == 3 and self.num_labels == 1:
                     threshold = np.expand_dims(threshold,[0,1])
                 elif nD == 2 and self.num_labels > 1:
                     threshold = np.expand_dims(threshold,[0,1])
                 elif nD == 2 and self.num_labels == 1:
                     threshold = np.expand_dims(threshold,[0])

                 
             else:
                 assert(False,"out_typ not implemented")
                 
                 
             if threshold is not None:
                 if out_typ.find('mask') != -1:
                     out_typ = 'uint8'   
                     if self.cropper.categorical:                     
                         tmp = np.argmax(res_,axis=-1)
                         if labelidx is not None:
                             pred_nii = nib.Nifti1Image(tmp==labelidx, newaffine, img1.header)
                         else:
                             pred_nii = nib.Nifti1Image(tmp, newaffine, img1.header)
                         
                     else:
                         if labelidx is not None:
                             pred_nii = nib.Nifti1Image(res_>threshold[...,labelidx:labelidx+1], newaffine, img1.header)
                         else:
                             pred_nii = nib.Nifti1Image(conncomp(res_>threshold), newaffine, img1.header)
                 if out_typ.find('atls') != -1 or out_typ == 'idx':
                     probs = None
                     if out_typ == 'idx':   # usually coming from QMembedding, first image is idxmap, second is probabilty
                         tmp = res_[...,0:1]
                         probs = res_[...,1:]
                         threshold_ = int(10000*ce_threshold)
                         tmp = tf.where(probs[...,0:1]<threshold_,0,tmp)
                        # probs = tf.where(probs<threshold_,0,probs)
                         tmp = np.int32(tmp)
                     elif self.cropper.categorical: # trained with CC-like loss, i.e. lastdim is of size 1+numberlabels
                         if votemap:
                             thevotemap = res_[...,-1:]
                             res_ = res_[...,0:-1]
                         tmp = np.expand_dims(np.argmax(res_,axis=-1),-1)
                         probs = np.max(res_,axis=-1,keepdims=True)
                         tmp = tf.where(probs<ce_threshold,0.0,tmp)
                         probs = np.int32( (probs * 10000) * np.int32(tmp>0))
                         if votemap:
                            probs = tf.concat([probs,thevotemap],-1)
                         tmp = np.int32(tmp)
                                                      
                     else:
                         tmp = res_>threshold
                         if len(tmp.shape) == nD:  # if it is already an index map, do nothing, just cast
                             tmp = tf.cast(res_,dtype=tf.int32)
                         else:  
                             if self.finalBlock is not None and hasattr(self.finalBlock,'name') and self.finalBlock.name == 'softmax0':
                                 # ccloss but with n channels, ccloss0
                                 res_ext = tf.concat([1-tf.reduce_sum(res_,-1,keepdims=True),res_],-1)
                                 tmp = np.int32(tf.expand_dims(np.argmax(res_ext,axis=-1),-1))
                                 probs = tf.expand_dims(np.int32( np.amax(res_ext,axis=-1) * 10000),-1) * np.int32(tmp>0)
                                 res = tmp                                 
                             else: # ordin. multiclass problem trained with BC, i.e. lastdim = numberlabels
                                 tmp = np.int32(tf.expand_dims((np.argmax(tmp*res_,axis=-1)+1)*(np.sum(tmp,axis=-1)>0),-1))
                                 probs = tf.expand_dims(np.int32( np.amax(res_,axis=-1) * 10000),-1) * np.int32(tmp>0)
                                 res = tmp
                             
                             
                     out_typ = 'int16'
                         
                     if self.cropper.categorial_label is not None: # of there is an indexmapping given, map back
                         idxmap = tf.cast([0] + self.cropper.categorial_label_original,dtype=tf.int32)
                         tmp = tf.gather(idxmap,tmp)
                         
                     tmp = conncomp(tmp,idx=self.cropper.categorial_label_original)
                         
                     if probs is not None:
                         tmp = tf.concat([tmp,probs],-1)
                         res = tmp
                     
                     if self.cropper.ndim==2 and squeeze:
                        tmp = np.squeeze(tmp)
                     
                     pred_nii = nib.Nifti1Image(tmp, newaffine, img1.header)
                     #j = KColormap.jet; str = ""; for (var k = 0; k < 256; k++) str += "[" +j[0][k] +"," +j[1][k] +"," +j[2][k] +"],"; 
                     if self.info is not None and 'xml' in self.info:
                        xml = self.info['xml'] + '   '
                     else:
                         colors = [[0,0,0],[0,0,135],[0,0,139],[0,0,143],[0,0,147],[0,0,151],[0,0,155],[0,0,159],[0,0,163],[0,0,167],[0,0,171],[0,0,175],[0,0,179],[0,0,183],[0,0,187],[0,0,191],[0,0,195],[0,0,199],[0,0,203],[0,0,207],[0,0,211],[0,0,215],[0,0,219],[0,0,223],[0,0,227],[0,0,231],[0,0,235],[0,0,239],[0,0,243],[0,0,247],[0,0,251],[0,0,255],[0,4,255],[0,8,255],[0,12,255],[0,16,255],[0,20,255],[0,24,255],[0,28,255],[0,32,255],[0,36,255],[0,40,255],[0,44,255],[0,48,255],[0,52,255],[0,56,255],[0,60,255],[0,64,255],[0,68,255],[0,72,255],[0,76,255],[0,80,255],[0,84,255],[0,88,255],[0,92,255],[0,96,255],[0,100,255],[0,104,255],[0,108,255],[0,112,255],[0,116,255],[0,120,255],[0,124,255],[0,128,255],[0,131,255],[0,135,255],[0,139,255],[0,143,255],[0,147,255],[0,151,255],[0,155,255],[0,159,255],[0,163,255],[0,167,255],[0,171,255],[0,175,255],[0,179,255],[0,183,255],[0,187,255],[0,191,255],[0,195,255],[0,199,255],[0,203,255],[0,207,255],[0,211,255],[0,215,255],[0,219,255],[0,223,255],[0,227,255],[0,231,255],[0,235,255],[0,239,255],[0,243,255],[0,247,255],[0,251,255],[0,255,255],[4,255,251],[8,255,247],[12,255,243],[16,255,239],[20,255,235],[24,255,231],[28,255,227],[32,255,223],[36,255,219],[40,255,215],[44,255,211],[48,255,207],[52,255,203],[56,255,199],[60,255,195],[64,255,191],[68,255,187],[72,255,183],[76,255,179],[80,255,175],[84,255,171],[88,255,167],[92,255,163],[96,255,159],[100,255,155],[104,255,151],[108,255,147],[112,255,143],[116,255,139],[120,255,135],[124,255,131],[128,255,128],[131,255,124],[135,255,120],[139,255,116],[143,255,112],[147,255,108],[151,255,104],[155,255,100],[159,255,96],[163,255,92],[167,255,88],[171,255,84],[175,255,80],[179,255,76],[183,255,72],[187,255,68],[191,255,64],[195,255,60],[199,255,56],[203,255,52],[207,255,48],[211,255,44],[215,255,40],[219,255,36],[223,255,32],[227,255,28],[231,255,24],[235,255,20],[239,255,16],[243,255,12],[247,255,8],[251,255,4],[255,255,0],[255,251,0],[255,247,0],[255,243,0],[255,239,0],[255,235,0],[255,231,0],[255,227,0],[255,223,0],[255,219,0],[255,215,0],[255,211,0],[255,207,0],[255,203,0],[255,199,0],[255,195,0],[255,191,0],[255,187,0],[255,183,0],[255,179,0],[255,175,0],[255,171,0],[255,167,0],[255,163,0],[255,159,0],[255,155,0],[255,151,0],[255,147,0],[255,143,0],[255,139,0],[255,135,0],[255,131,0],[255,128,0],[255,124,0],[255,120,0],[255,116,0],[255,112,0],[255,108,0],[255,104,0],[255,100,0],[255,96,0],[255,92,0],[255,88,0],[255,84,0],[255,80,0],[255,76,0],[255,72,0],[255,68,0],[255,64,0],[255,60,0],[255,56,0],[255,52,0],[255,48,0],[255,44,0],[255,40,0],[255,36,0],[255,32,0],[255,28,0],[255,24,0],[255,20,0],[255,16,0],[255,12,0],[255,8,0],[255,4,0],[255,0,0],[251,0,0],[247,0,0],[243,0,0],[239,0,0],[235,0,0],[231,0,0],[227,0,0],[223,0,0],[219,0,0],[215,0,0],[211,0,0],[207,0,0],[203,0,0],[199,0,0],[195,0,0],[191,0,0],[187,0,0],[183,0,0],[179,0,0],[175,0,0],[171,0,0],[167,0,0],[163,0,0],[159,0,0],[155,0,0],[151,0,0],[147,0,0],[143,0,0],[139,0,0],[135,0,0],[131,0,0],[128,0,0]]
                         #colors = [[255,0,0],[0,255,0],[0,0,255],[255,255,0],[255,0,255],[0,255,255],[255,128,0],[255,0,128],[128,255,128],[0,128,255],[128,128,128],[185,170,155]]
                         xmlpre = '<?xml version="1.0" encoding="UTF-8"?> <CaretExtension>  <Date><![CDATA[2013-07-14T05:45:09]]></Date>   <VolumeInformation Index="0">   <LabelTable>'
                         body = ''
                         
                         num_labels = self.num_labels
                         
                         if self.cropper.categorical:                         
                             num_labels = self.num_labels-1
                         for k in range(num_labels):
                            key = k+1
                            if self.cropper.categorial_label is not None:
                                key = self.cropper.categorial_label_original[k]
                            
                            if label_names is None:
                                labelname = "L" + str(k+1)
                                if self.cropper.categorial_label is not None:
                                    if k+1 != self.cropper.categorial_label_original[k]:
                                        labelname = "L" + str(k+1) +"_"+ str(self.cropper.categorial_label_original[k])
                                        
                            else:
                                labelname = label_names[k] + "-" + str(k+1)
                            if label_colors is None:     
                                rgb = colors[(key*32)%256]
                                #rgb = colors[k%len(colors)]
                            else:
                                rgb = label_colors[k]
                            print("writing xml header")
                            body += '<Label Key="{}" Red="{}" Green="{}" Blue="{}" Alpha="1"><![CDATA[{}]]></Label>\n'.format(key,rgb[0]/255,rgb[1]/255,rgb[2]/255,labelname)
                         xmlpost = '  </LabelTable>  <StudyMetaDataLinkSet>  </StudyMetaDataLinkSet>  <VolumeType><![CDATA[Label]]></VolumeType>   </VolumeInformation></CaretExtension>'
                         xml = xmlpre + "\n" + body + "\n" + xmlpost + "\n               \n                "
                     
                     pred_nii.header.extensions.clear()
                     pred_nii.header.extensions.append(nib.nifti1.Nifti1Extension(0,bytes(xml,'utf-8')))

                     
                 pred_nii.header.set_slope_inter(1,0)
             else:    
             
                 if self.cropper.ndim==2 and squeeze:
                     res_ = np.squeeze(res_)
                
                 pred_nii = nib.Nifti1Image(res_*fac, newaffine, img1.header)
                 pred_nii.header.set_slope_inter(1/(0.000001+fac),0.0000000)

             pred_nii.header.set_data_dtype(out_typ)                     
             pred_nii.header['cal_max'] = 1
             pred_nii.header['cal_min'] = 0
             pred_nii.header['glmax'] = 1
             pred_nii.header['glmin'] = 0
             pred_nii.header['descrip'] = 'hlim:[0,1]'

             if name is not None:
                 nib.save(pred_nii,name)
             return pred_nii
                        
          if isinstance(ofname,list):
             if len(ofname) == 1:
                 pred_nii = savenii(ofname[0],res,out_typ)
             else:
                 for s in range(len(ofname)):                    
                     pred_nii = savenii(ofname[s],res[...,s],out_typ,s)
          else:
             pred_nii = savenii(ofname,res,out_typ)
              
      print('done')            
      if pred_nii is not None:        
          if return_nibabel:
              if maxidxperlabel is not None:
                  return pred_nii,res,maxidxperlabel
              else:
                  return pred_nii,res
          else:
              if maxidxperlabel is not None:              
                  return newaffine,res,maxidxperlabel
              else:
                  return newaffine,res
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
           clsCreator=None ,dummyData=None    ):

    custom_objects = custom_layers

    name = name.replace(".json","")
    fname = name + ".json"
    with open(fname) as f:
        x = json.load(f)
    
    if 'cropper' not in x:
        raise(RuntimeError('not a valid patchwork model json'))
        os._exit(127)


    if 'system' not in x['cropper']:
        print('deprecated: no system variable found, using system=matrix')
        x['cropper']['system'] = 'matrix'

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
            model.load_weights(weightspath).expect_partial()    
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
                        
            model.load_weights(loading_name_tmp + suffix).expect_partial()                
        
        
        try:
            if dummyData is None:
                if model.cropper.ndim == 3:
                    initdat = tf.ones([1,320,320,320, model.input_fdim])
                else:
                    initdat = tf.ones([1,32,32, model.input_fdim])    
            else:
                initdat = dummyData
            print("----------------- load/init network by minimal application")
            dummy = model.apply_full(initdat,resolution={"input_edges":np.eye(4),"voxsize":[0.01,0.01,0.01]},
                                     sampling_factor=0.2,
                                     verbose=False,scale_to_original=False,generate_type='random',repetitions=1,init=True)        
            print("----------------- model and weights loaded")
        except:
            if not notmpfile:
                shutil.rmtree(tmpdir)
            raise 
        
        if not notmpfile:
            shutil.rmtree(tmpdir)
    else:        
        model.load_weights(weightspath).expect_partial()       


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
  #   balance -  a dict {'ratio':r} , where r gives desired balance between
  #         positive and negative examples
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
            num_patches_to_train=None,
            traintype='random',
            lazyTrain=None,
            num_patches=10,
            
            train_ids = None,
            unlabeled_ids = [],
            valid_ids = [],
            valid_num_patches=None,
            
            hard_mining = 0,
            hard_mining_maxage=50,
            hard_mining_order='loss',
                        
            shuffle_buffer_size=1000,
            self_validation=False,
            max_depth=None,
            depth_schedule=None,
            batch_size=32,
            verbose=1,
            debug=False,
            steps_per_epoch=None,
            inc_train_cycle=True,
            cp_interval=-1,
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
            recompile_loss_optim=False,
            dontcare=True,
            report_perf=True,
            patch_on_cpu=True,
            parallel=False,
            fit_type='custom',
            train_S = True,
            train_U = True,
            train_D = True,
            train_space=False,
            callback=None
            ):

    if not train_space:
        train_space = self.space_loss is not None
      
    import logging
    logging.getLogger('tensorflow').setLevel(logging.ERROR)      
    def f1_metric(y_true, y_pred,from_logits=False,start_reduce=0):
        y_true = tf.where(y_true>0.0,1.0,0.0)
        
        sz = y_true.shape
        def sumy(x):
            return tf.reduce_sum(x,axis=list(range(start_reduce,self.cropper.ndim+1)))
        
        if from_logits:
            y_pred = tf.nn.sigmoid(y_pred)
        true_positives = sumy(kb.round(kb.clip(y_true * y_pred, 0, 1)))
        false_positives = sumy(kb.round(kb.clip((1-y_true) * y_pred, 0, 1)))
        possible_positives = sumy(kb.round(kb.clip(y_true, 0, 1)))
        predicted_positives = sumy(kb.round(kb.clip(y_pred, 0, 1)))


        f1_val = (2*true_positives+1)/(possible_positives+predicted_positives+2)
#        f1_val = (true_positives)/(possible_positives+predicted_positives+1)
        return f1_val      
      
    def f1_metric_best(y_true, y_pred,from_logits=False):
        y_true = tf.where(y_true>0.0,1.0,0.0)
        
        if from_logits:
            y_pred = tf.nn.sigmoid(y_pred)
        
        sz = y_pred.shape
        y_pred = tf.reshape(y_pred,[tf.reduce_prod(sz)])
        y_true = tf.reshape(y_true,[tf.reduce_prod(sz)])
        idx = tf.argsort(y_pred,0,'DESCENDING')
        
        pval = tf.gather(y_pred,idx)
        ranking = tf.gather(y_true,idx)
        
        TP = tf.cumsum(ranking)
        predicted_positives = tf.range(1,(ranking.shape[0])+1,dtype=self.dtype)
        possible_positives = tf.reduce_sum(ranking)
        f1_val = (2*TP+1)/(possible_positives+predicted_positives+2)
       #  precision = TP / (kb.epsilon()+tf.range(1,(ranking.shape[0])+1,dtype=self.dtype))
       #  recall = TP / (kb.epsilon()+tf.reduce_sum(ranking))
       #  f1_val = 2*(precision*recall)/(precision+recall+kb.epsilon())

            
                
        best_idx = tf.argmax(f1_val)
        f1 = f1_val[best_idx]
        th = pval[best_idx]
                    
        return f1,th

    def getSample(subset,np,valid=False,lazyEval=None,skipLabels=False):
        tset = [trainset[i] for i in subset]
        lset=None
        if len(labelset) > 0:
            lset = [labelset[i] for i in subset]      
        if skipLabels:
            lset=None
        rset = None
        if resolutions is not None:
            rset = [resolutions[i] for i in subset]      

        
        dphi = rot_intrinsic
        aug_ = augment
        if valid:
            dphi=0
            aug_={'dphi':0,'dscale':0,'flip':0}

        branch_factor = 1
        if lazyEval is not None and "branch_factor" in lazyEval:
            branch_factor = lazyEval['branch_factor']
        
        self.cropper.dest_full_size = [None]*self.cropper.depth

        
            
        with tf.device(DEVCPU):    
    
            if traintype == 'random' or traintype ==  'random_deprec' :
                c = self.cropper.sample(tset,lset,resolutions=rset,generate_type=traintype,max_depth=max_depth,
                                        num_patches=np,augment=aug_,balance=balance,dphi=dphi,lazyEval=lazyEval,branch_factor=branch_factor,createCoordinateLabels=train_space,training=True)
            elif traintype == 'tree':
                c = self.cropper.sample(tset,lset,resolutions=rset,generate_type='tree_full', jitter=jitter,max_depth=max_depth,
                                        jitter_border_fix=jitter_border_fix,augment=aug_,balance=balance,dphi=dphi,branch_factor=branch_factor,createCoordinateLabels=train_space,training=True)
            
        
        return c
    
    
    def computeloss(fun,label,prediction):
        
        def theloss(pred):
            if self.cropper.categorial_label is not None and not self.cropper.categorical and not hasattr(pred,'QMembedding'):
                lmat = 0.0
                cnt = 0
                for j in self.cropper.categorial_label:
                    lmat += fun(tf.cast(label==j,dtype=tf.float32),pred[...,cnt:cnt+1])
                    cnt=cnt+1
                lmat = lmat/cnt
            else:  
                if self.num_labels != -1 and not hasattr(pred,'QMembedding'):
                        pred = pred[...,0:self.num_labels]
                lmat = fun(label,pred)
            return lmat

        if hasattr(prediction,'deep_out'):
            loss = theloss(prediction)
            for k in range(len(prediction.deep_out)):                
               loss = loss + theloss(prediction.deep_out[k])
            return loss
        else:
            return theloss(prediction)
    
    def computeF1perf(label,pred,from_logits=False):
        f1=[]
        th=[]
        cnt = 0
        if self.cropper.categorial_label is not None:
            if hasattr(pred,'QMembedding'):
                predQM = pred.QMembedding.apply(pred)
                predQM = predQM[...,0:1]
            for j in self.cropper.categorial_label:
                if hasattr(pred,'QMembedding'):
                    f1_ = tf.reduce_mean(f1_metric(tf.cast(label==j,dtype=tf.float32),tf.cast(predQM==j,tf.float32),from_logits=from_logits))
                    th_ = tf.cast(0.5,tf.float32)
                else:
                    f1_,th_ = f1_metric_best(tf.cast(label==j,dtype=tf.float32),pred[...,cnt:cnt+1],from_logits=from_logits)
                f1.append(f1_)
                th.append(th_)
                cnt=cnt+1
        else:                       
            for j in range(0,min(self.num_labels,label.shape[-1],pred.shape[-1])  )  :
                f1_,th_ = f1_metric_best(tf.cast(label[...,j:j+1],dtype=tf.float32),pred[...,j:j+1],from_logits=from_logits)
                f1.append(f1_)
                th.append(th_)
                cnt=cnt+1
                
        if self.cropper.categorical:                
            meanf1 = sum(f1[1:])/(cnt-1)
        else:
            meanf1 = sum(f1)/cnt
                
        return f1,th,meanf1

    def computeF1perf_center(label,pred,from_logits=False):
        f1=0
        th=0
        cnt = 0
        if self.cropper.categorial_label is not None:
            for j in self.cropper.categorial_label:
                f1_ = f1_metric(tf.cast(label==j,dtype=tf.float32),pred[...,cnt:cnt+1],from_logits=from_logits,start_reduce=1)
                f1+=f1_
                cnt=cnt+1
        else:                       
            for j in range(0,self.num_labels):
                f1_ = f1_metric(tf.cast(label[...,j:j+1],dtype=tf.float32),pred[...,j:j+1],from_logits=from_logits,start_reduce=1)
                f1+=f1_
                cnt=cnt+1
        f1 /= cnt
        return f1
    
    def valid_step_supervised(images,lossfun,prefix):
            
      hist = {}
      
      data = images[0]
      labels = images[1]      
     
      preds = self(data, training=True)
      loss = 0
      for k in range(len(labels)):          
            if dontcare:
                if self.cropper.categorial_label is not None:
                    masked_pred = tf.where(labels[k]==-1,tf.cast(0,preds[k].dtype),preds[k])
                    masked_label = tf.where(labels[k]==-1,tf.cast(0,labels[k].dtype),labels[k])
                else:                
                    masked_pred = tf.where(tf.math.is_nan(labels[k]),0.0,preds[k][...,0:labels[k].shape[-1]])
                    masked_label = tf.where(tf.math.is_nan(labels[k]),0.0,labels[k])
            else:
                masked_pred = preds[k]
                masked_label = labels[k]
            l = computeloss(lossfun[k],masked_label,masked_pred)            
            l = tf.reduce_mean(l)
            loss += l
            if k == len(labels)-1:
                hist[prefix+'_output_'+str(k+1)+'_loss'] = l
                f1list,th,f1 = computeF1perf(masked_label,masked_pred,from_logits=self.finalizeOnApply)
                hist[prefix+'_output_'+str(k+1)+'_f1'] = 10**f1
                hist[prefix+'_output_'+str(k+1)+'_threshold'] = 10**(sum(th)/len(th))
                hist[prefix+'_nodisplay_class_f1'] = tf.cast(f1list,dtype=tf.float32)
                hist[prefix+'_nodisplay_class_threshold'] = tf.cast(th,dtype=tf.float32)
                
      return hist
 
    
  
    
    def relDistanceLoss(x,y,level,full='dummy',typ='patchwise',weight=1,d0=500,s0=50):
        sz = x.shape
        n = tf.reduce_prod(sz[1:-1])
        bsz = sz[0]
        x=tf.reshape(x,[-1,x.shape[-1]])
        y=tf.reshape(y,[-1,y.shape[-1]])
        indices = tf.range(start=0, limit=x.shape[0], dtype=tf.int32)
        shuffled_indices = tf.random.shuffle(indices)
    
        if typ == 'patchwise':
            offset = tf.repeat(tf.range(start=0, limit=bsz, dtype=tf.int32)*n,n)    
            shuffled_indices = tf.math.mod(shuffled_indices,n) + offset
        
        
        shuffled_x = tf.gather(x, shuffled_indices)
        shuffled_y = tf.gather(y, shuffled_indices)  
        l = tf.einsum('xy,xy->x',shuffled_y,y)+self.space_offset[level]*100
        d2 = tf.math.sqrt(tf.reduce_sum((shuffled_x-x)**2,-1))
        t = tf.math.sigmoid( -(d2-d0)/s0 )
        return weight*tf.keras.losses.binary_crossentropy (t,l,from_logits=True)
   
    def train_step_supervised(images,lossfun):
            
      hist = {}
      
      data = images[0]
      labels = images[1]
      
      trainvars = self.block_variables
      if self.finalBlock is not None:
          trainvars = trainvars + self.finalBlock.trainable_variables
      if hasattr(self,'external_trainvars'):
          trainvars = trainvars + self.external_trainvars

      with tf.GradientTape() as tape:
        preds = self(data, training=True)
            
        loss = 0
        depth = min(len(labels),self.cropper.depth)
        for k in range(depth):
            if lossfun[k] is not None:
                if dontcare:
                    if self.cropper.categorial_label is not None:
                        masked_pred = tf.where(labels[k]==-1,tf.cast(0,preds[k].dtype),preds[k])
                        masked_label = tf.where(labels[k]==-1,tf.cast(0,labels[k].dtype),labels[k])
                    else:
                        masked_pred = tf.where(tf.math.is_nan(labels[k]),0.0,preds[k][...,0:labels[k].shape[-1]])                                         
                        masked_label = tf.where(tf.math.is_nan(labels[k]),0.0,labels[k])
                else:
                    masked_pred = preds[k]
                    masked_label = labels[k]
                
                if len(masked_label.shape) > 2: # then it is not a dummy
                    lmat = computeloss(lossfun[k],masked_label,masked_pred)
                    l = tf.reduce_mean(lmat)
                    loss += l
                    
                    if k == depth-1:# or depth_schedule is not None:
                        from_logits = self.finalizeOnApply or (k < self.cropper.depth-1)
                        hist['output_' + str(k+1) + '_loss'] = l      
                        if report_perf:    
                            f1list,th,f1 = computeF1perf(masked_label,masked_pred,from_logits=from_logits)                                        
                            hist['output_' + str(k+1) + '_f1'] = 10**f1
                            hist['output_' + str(k+1) + '_threshold'] = 10**(sum(th)/len(th))
                            hist['nodisplay_class_f1'] = tf.cast(f1list,dtype=tf.float32)
                            hist['nodisplay_class_threshold'] = tf.cast(th,dtype=tf.float32)
                                                                                       
                        if hard_mining > 0:
                            order = lmat
                            if hard_mining_order=='f1':
                                order = computeF1perf_center(masked_label,masked_pred,from_logits=from_logits)
                                
                            if len(order.shape) < self.cropper.ndim+1:
                                hist['loss_per_patch'] = tf.reduce_mean(order,axis=1)
                            else:
                                hist['loss_per_patch'] = tf.reduce_mean(order,axis=list(range(1,self.cropper.ndim+1)))
                        
                if train_space:
                    
                    if self.space_loss is not None and 'full' in self.space_loss and self.space_loss['full']:
                        l = relDistanceLoss(labels[k+depth],preds[k],k,**self.space_loss)
                    elif k < self.cropper.depth-1:
                        l = relDistanceLoss(labels[k+depth],preds[k][...,self.num_labels:],k,**self.space_loss)
                    loss += l
                  #  if k == depth-1:
                    hist['output_' + str(k+1) + '_spc'] = l

            
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
        
    batch_size_intended = batch_size
    
    if self.saved_points is None:
        self.saved_points = {}
        
    if self.train_cycle is None:
       self.train_cycle = 0   
    if inc_train_cycle:
        self.train_cycle += 1
        
    
    if isinstance(augment,dict):        
        self.augment = augment
    
    
    if valid_num_patches is None:
        valid_num_patches = num_patches
 
    if recompile_loss_optim or (not hasattr(self,'optimizer') or self.optimizer is None):    
        if optimizer is None:
            optimizer = tf.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=True)
        else:
            print('taking custom optimizer')
        self.optimizer = optimizer            


    def createLossArray(lossfun):
        loss = []
        if self.intermediate_loss:
            for k in range(self.cropper.depth-1):
                loss.append(lambda x,y: lossfun(x,y,from_logits=True))
        if self.finalizeOnApply:
            loss.append(lambda x,y: lossfun(x,y,from_logits=True))
        else:
            loss.append(lambda x,y: lossfun(x,y,from_logits=False))
        return loss
        


    if recompile_loss_optim or not hasattr(self,'loss'):
        if loss is None:
            print("using default bc loss")
            self.loss = createLossArray(tf.keras.losses.binary_crossentropy)
        else:
            if callable(loss):
                print("creating custom loss array")                
                self.loss = createLossArray(loss)
            else:            
                print("using custom loss array")
                self.loss = loss
    
    loss = self.loss
    
    
    trained_blocks = len(self.blocks)
    if max_depth is not None:
        trained_blocks = max_depth
    
    self.block_variables = list([])
    for b in range(trained_blocks):
        self.block_variables += self.blocks[b].trainable_variables
        
    self.disc_variables = list([])
    for b in range(trained_blocks):
        if b < len(self.classifiers):
            self.disc_variables += self.classifiers[b].trainable_variables
    
    

    
    if self.cropper.categorial_label is not None:

       if self.cropper.categorical:
           nl = len(self.cropper.categorial_label_original)
           self.cropper.categorial_label = list(range(0,nl+1))
           c = tf.cast([0] + self.cropper.categorial_label_original,dtype=tf.int64)       
           r = tf.range(0,nl+1)
           self.num_labels=nl+1
           self.cropper.num_labels = nl+1
       else:
           self.cropper.categorial_label = list(range(1,self.num_labels+1))
           c = tf.cast(self.cropper.categorial_label_original,dtype=tf.int64)
           r = tf.range(1,self.num_labels+1)
       for k in range(len(labelset)):
            maxi  = max(c) # tf.reduce_max(labelset[k])
            #maxi  = tf.reduce_max(labelset[k])+
            
            dontcarelabel = labelset[k]==-1

            
            labelset[k] = tf.where(dontcarelabel,0,labelset[k])
            labelset[k] = tf.where(labelset[k]>tf.cast(maxi,dtype=labelset[k].dtype),0,labelset[k])
            categorial_label_idxmap=  tf.scatter_nd( tf.expand_dims(c,1), r, [maxi+1])      
            labelset[k] = tf.expand_dims(tf.gather_nd(categorial_label_idxmap,tf.cast(labelset[k],dtype=tf.int32)),-1)
            labelset[k] = tf.where(dontcarelabel,-1,labelset[k])


    if self.cropper.categorial_label_original is not None:               
        if self.cropper.categorical:
            self.myhist.categorial_label = [0] + self.cropper.categorial_label_original
        else:
            self.myhist.categorial_label = self.cropper.categorial_label_original
    
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
            
    
    
    if lazyTrain is not None:

  
        
        scalarLabels = False
        
        if len(labelset[trainidx[0]].shape) == 2:
            scalarLabels = True
        
        
        trainvars = self.block_variables
        trainvars = trainvars + self.finalBlock.trainable_variables


        def trainalt_step(i):
                with tf.GradientTape() as tape:                    
                  losslog = {}
                  theloss = [0]*self.cropper.depth
                  total = 0
                  for k in range(len(trainidx)):        
                      print(".",end="" )                      
                      c_data = getSample([trainidx[k]],num_patches,lazyEval=lazyTrain,
                                                                   skipLabels=scalarLabels)    
                      preds = self(c_data.getInputData(), training=True,lazyEval=lazyTrain)
                      for i in range(len(preds)):
                          if scalarLabels:
                              p = tf.expand_dims(tf.expand_dims(tf.reduce_max(preds[i]),0),1)
                              theloss[i] += loss[i](labelset[trainidx[k]],p)
                          else:
                              p = preds[i]
                              theloss[i] += loss[i](t[i],p)
                          total = total + theloss[i]
                    
                  for i in range(len(theloss)):
                     losslog.update({'out_'+str(i) : tf.reduce_mean(theloss[i])/len(trainidx) })
                                         
                            
                gradients = tape.gradient(total,trainvars)
                self.optimizer.apply_gradients(zip(gradients, trainvars))

                return losslog

        # if (not "trainalt_step" in self.compiled) or recompile_loss_optim:
        #     if debug:
        #         self.compiled["trainalt_step"] = trainalt_step
        #     else:
        #         self.compiled["trainalt_step"] = tf.function(trainalt_step)


        for i in range(num_its):
            start = timer()
            losslog = trainalt_step(i)
            
            self.myhist.accum('train',[losslog],len(trainidx),tensors=True,mean=True)
            self.trained_epochs+=len(trainidx)

            print("")
            for k in losslog:
                    print(k + ":" + str(losslog[k].numpy()),end=" ")            
            
            print("elapsed %.3fs   time per image: %.3fs"%((timer()-start),(timer()-start)/len(trainidx)),flush=True)

            if autosave:
               if self.modelname is None:
                   print("no name given, not able to save model!")
               else:
                   self.save(self.modelname)
    
            if showplot:
                self.myhist.show_train_stat()
                
            if callback is not None:
                callback(i)


    else:
        was_last_min = False
        
        if hard_mining is not None and hard_mining>0 and hasattr(self,'hard_data'):
            hard_data = self.hard_data
        else:
            hard_data = None
            
        epochs_ = epochs
            
        if depth_schedule is not None:
            recompile_loss_optim = True
    
        initial_number_of_trained_patches = self.myhist.getNumberOfSeenPatches()
    
            
        if parallel:
            if parallel == True:
                parallel = 'thread'
            worker = PatchWorker(self,parallel,
                                      {'trainidx':trainidx,'trainset':trainset, 'labelset':labelset, 'resolutions':resolutions,
                                       'traintype':traintype, 'max_depth':max_depth,
                                       'augment':augment,'num_patches':num_patches, 'balance':balance,
                                       'jitter':jitter, 'jitter_border_fix':jitter_border_fix
                                       })
        for i in range(num_its):
    
            if depth_schedule == 1:
                #max_depth=3
                max_depth=((i+self.cropper.depth-1)%self.cropper.depth)+1
                #max_depth=((i)%self.cropper.depth)+1
                #epochs=(self.cropper.depth+1-max_depth)*epochs_
    
            print("------- train depth:" + str(max_depth))
    
            print("----------------------------------------- iteration:" + str(i))
            
            ### sampling
            if parallel:
                print("getting patches from patchworker")
                start = timer()
                c_data = worker.getData()
                print("received patches in  %.3fs"%(timer()-start),flush=True)
            else:            
                print("sampling patches for training")
                start = timer()
                c_data = getSample(trainidx,num_patches)    
                end = timer()
                ratio = (end-start)/(len(trainidx)*num_patches)*1000
                print("sampled " + str(len(trainidx)*num_patches) + " patches with %.2f ms/sample"%ratio,flush=True)
    
            pixelratio,pixelfreqs = self.cropper.computeBalances(c_data.scales,True,balance)        
            self.pixelfreqs = pixelfreqs
            self.myhist.pixelratio = pixelratio
            
            
            if hard_data is not None:
               with tf.device(DEVCPU):              
                   print("balance hard data")
                   self.cropper.computeBalances(hard_data.scales,True,balance)               
                   c_data.merge(hard_data)
            
            total_numpatches = c_data.num_patches()
            if batch_size_intended > total_numpatches:
                batch_size = total_numpatches        
            else:
                batch_size = batch_size_intended
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
                
    
            print("starting training")
            start = timer()
            
            lam = tf.convert_to_tensor(0.1)
    
    
            if fit_type == 'custom':
                shuffle_buffer = max(shuffle_buffer_size,total_numpatches)
                dataset = c_data.getDataset(sampletyp=sampletyp).shuffle(shuffle_buffer).batch(batch_size,drop_remainder=True)
    
                if (not "train_step" in self.compiled) or recompile_loss_optim:
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
    
                if (not "train_step_discrim" in self.compiled) or recompile_loss_optim:
                    if debug:
                        self.compiled["train_step_discrim"] = (train_step_discriminator)
                        self.compiled["train_step_unsuper"] = (train_step_unsupervised)
                    else:
                        self.compiled["train_step_discrim"] = tf.function(train_step_discriminator)
                        self.compiled["train_step_unsuper"] = tf.function(train_step_unsupervised)
                
                patchloss = []
                
                
                actual_epochs = epochs            
                if parallel:
                    actual_epochs = 1000             
                for e in range(actual_epochs):
                    
                    if parallel:
                        #if worker.queue.qq.full() and e >= epochs:
                        if worker.queue.full() and e >= epochs:
                            break
    
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
                            if 'loss_per_patch' in losslog:
                                patchloss.append(tf.concat([tf.expand_dims(tf.cast(labeled_element[0]['batchindex'],dtype=tf.float32),1),
                                                            tf.expand_dims(losslog['loss_per_patch'],1)],1))
                                del losslog['loss_per_patch']
                            
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
                        
                    
                    
                    log, new_min = self.myhist.accum('train',log,numsamples,tensors=True,mean=True)
                    self.trained_epochs+=numsamples
    #                log, new_min = self.myhist.accum('train',log,1,tensors=True,mean=True)
     #               self.trained_epochs+=1
                    end = timer()                
                    print( "  %.2f ms/sample " % (1000*(end-sttime)/(numsamples+numsamples_unl)))
                    for k in log:
                        if len(log[k][0].shape)==0 or log[k][0].shape[0] < 10 :
                            print(k + ":" + str(log[k][0]),end=" ")
                        else:
                            print(k + ": mean=" + str( np.mean(log[k][0])),end=" ")
                    print("")
                    if hasattr(self.optimizer,'_decayed_lr'):
                        print("learning rate: " + str(self.optimizer._decayed_lr(tf.float32).numpy()))
    
                if len(patchloss) > 0:
                    patchloss = tf.concat(patchloss,0)
                    patchloss = tf.scatter_nd(tf.cast(patchloss[:,0:1],dtype=tf.int32),patchloss[:,1:2],[total_numpatches,1])
    
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
            
            if hard_mining is not None and hard_mining>0:
               with tf.device(DEVCPU):    
                  hard_mining_ratio = 1-1/(1+hard_mining)
                  if hard_mining_order == 'balance':
                      targetdata = c_data.getTargetData(sampletyp)
                      patches = targetdata[-1]
                      labelfreqs = []
    
                      for k in range(self.num_labels):
                          if self.cropper.categorial_label is not None:
                                labelfreqs.append(tf.reduce_max(tf.cast(patches==self.cropper.categorial_label[k],dtype=tf.float32),axis=range(1,self.cropper.ndim+1)))
                          else:
                                labelfreqs.append(tf.reduce_max(tf.cast(patches[...,k:k+1],dtype=tf.float32),axis=range(1,self.cropper.ndim+1)))
                          
                      labelfreqs = tf.concat(labelfreqs,1)
                      labelfreqs = labelfreqs / (0.00001+tf.reduce_sum(labelfreqs,axis=0))
                      probs = tf.reduce_max(labelfreqs,axis=1,keepdims=True)
                      probs = tf.concat([probs,tf.expand_dims(c_data.getAge(),1)],1)
                      c_data.subsetProb(probs,hard_mining_ratio,hard_mining_maxage)    
                      
                  else:
                      patchloss = tf.concat([patchloss,tf.expand_dims(c_data.getAge(),1)],1)                                          
                      c_data.subsetOrder(patchloss,hard_mining_ratio,hard_mining_maxage)    
                  self.myhist.age = c_data.getAge()
                  hard_data = c_data
                  self.hard_data = hard_data
            else:
               del c_data.scales
               del c_data
               if hasattr(self.myhist,'age'):
                   del self.myhist.age
               c_data = None
    
            
                        
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
    
            if cp_interval != -1:
                if (i+1)%cp_interval == 0:
                    self.train_cycle += 1
            
            if autosave:
               if self.modelname is None:
                   print("no name given, not able to save model!")
               else:
                   self.save(self.modelname)
    
            if showplot:
                self.myhist.show_train_stat()
                
            if num_patches_to_train is not None:
               if self.myhist.getNumberOfSeenPatches()-initial_number_of_trained_patches > num_patches_to_train:
                   break
                
    
    
        if parallel:        
            worker.kill()

            
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
        if isinstance(obj,tf.TensorShape):
           return list(obj)            
        if isinstance(obj,list):
           return list(obj)
        if hasattr(obj,'tolist'):
           return obj.tolist()
        return json.dumps(obj,cls=patchworkModelEncoder)
        
