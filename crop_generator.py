#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 15:37:03 2020

@author: reisertm
"""

import tensorflow as tf
from . improc_utils import *
from timeit import default_timer as timer
from collections.abc import Iterable

########## Cropmodel ##########################################


class CropInstanceLazy:
  def __init__(self,localcrop,cropper):
      self.cropper = cropper
      self.localcrop = localcrop
      self.lastscale = None
      self.level = 0
      self.scales = []

  # get input training data 
  def getInputData(self):
      def getNext(idx=None):          
          if idx is not None:
              self.lastscale = self.cropper.take_subselection(self.lastscale,idx)
          self.lastscale = self.localcrop(self.lastscale,self.level)
          self.scales.append(self.lastscale)
          self.level = self.level + 1
          return self.lastscale['data_cropped'],self.lastscale['local_box_index']
      return getNext
            
  def stitchResult(self,r,level):
     return stitchResult(r,level,self.scales,self.cropper.scatter_type)




class CropInstance:
  def __init__(self,scales,cropper,intermediate_loss):
    self.scales = scales
    self.cropper = cropper
    self.intermediate_loss = intermediate_loss

  def extb2dim(self,x,batchdim2):
      if batchdim2 == -1:
          return x      
      sz = x.shape
      x = tf.reshape(x,[sz[0]//batchdim2, batchdim2] + sz[1:])
      return x

  # get input training data 
  def getInputData(self,batchdim2=-1):                    
    cnt = 0
    inp = {}
    for x in self.scales:
      inp['input'+str(cnt)] = self.extb2dim(x['data_cropped'],batchdim2)
      inp['cropcoords'+str(cnt)] = self.extb2dim(x['local_box_index'],batchdim2)
      cnt = cnt + 1
    return inp

  # get target training data 
  def getTargetData(self,batchdim2=-1):
    out = []
    
    create_indicator_classlabels = self.cropper.create_indicator_classlabels
    classifier_train = self.cropper.model.classifier_train
    spatial_train = self.cropper.model.spatial_train
    intermediate_loss = self.cropper.model.intermediate_loss
    cls_intermediate_loss= self.cropper.model.cls_intermediate_loss
    depth = len(self.scales)
    for i in range(depth-1):
        x = self.scales[i]
        if cls_intermediate_loss and classifier_train:
            out.append(x['class_labels'])            
        if intermediate_loss and spatial_train:
            out.append(x['labels_cropped'])
        
    if classifier_train or self.cropper.model.spatial_max_train:
        out.append(self.scales[-1]['class_labels'])
    if spatial_train and not self.cropper.model.spatial_max_train:
        out.append(self.scales[-1]['labels_cropped'])
    
    if batchdim2 != -1:
        for k in range(len(out)):
            out[k] = self.extb2dim(out[k],batchdim2)
            out[k] = tf.reduce_max(out[k],axis=1)
        
    return out

  def stitchResult(self,r,level):
     return stitchResult(r,level,self.scales,self.cropper.scatter_type)

# stitched results (output of network) back into full image
def stitchResult(r,level, scales,scatter_type):
    qq = r[level]
    numlabels = qq.shape[-1]
    sc = scales[level]
    pbox_index = sc['parent_box_scatter_index']
    sha = list(sc['dest_full_size'])
    sha.append(numlabels)
    sha = sha[1:]
    
    if scatter_type=='NN':        
        return tf.scatter_nd(pbox_index,qq,sha), tf.scatter_nd(pbox_index,qq*0+1,sha);
    else:
        return scatter_interp(pbox_index,qq,sha)


def scatter_interp(x,data,sz):
    
    
      # sz = []
      # for i in range(len(sz_)-1):
      #     sz.append(sz_[i]+10)
      # sz.append(sz_[-1])
    
      nD = len(sz)-1
      
      ones = data*0+1
      
      def frac(a):
          return a-tf.floor(a)
      if nD == 3:
          w = [frac(x[:,0:1,0:1,0:1, 0:1]),frac(x[:,0:1,0:1,0:1, 1:2]), frac(x[:,0:1,0:1,0:1, 2:3])]
      else:
          w = [frac(x[:,0:1,0:1, 0:1]),frac(x[:,0:1,0:1, 1:2]) ]

      def stitch(d,idx,s):
          q = tf.convert_to_tensor(s)
          for k in range(nD+1):
              q = tf.expand_dims(q,0)
          idx = idx + q
          weight = 1.0
          for k in range(nD):
              if s[k] == 1:
                  weight = weight*w[k]
              else:
                  weight = weight*(1-w[k])
          res_ = 0
          sums_ = 0
          for i in range(idx.shape[0]):
               res_ = res_ + weight[i]*tf.scatter_nd(idx[i,...] , data[i,...], sz)
               sums_ = sums_ + weight[i]*tf.scatter_nd(idx[i,...] , ones[i,...], sz)
               
          return res_,sums_
      x = tf.cast(x,dtype=tf.int32)
      if nD == 3:
          ids = [[0,0,0],[0,0,1],[0,1,0],[1,0,0],[0,1,1],[1,0,1],[1,1,0],[1,1,1]]
      else:
          ids = [[0,0],[0,1],[1,0],[1,1]]
      res = 0
      sums = 0
      for k in ids:
          r,s = stitch(data,x,k)
          res = res + r
          sums = sums + s
      return res,sums


class CropGenerator():

  def __init__(self,patch_size = (64,64),  # this is the size of our patches
                    scale_fac = 0.4,       # scale factor from one scale to another
                    scale_fac_ref = 'max', # for inhomgenus matrixsizes scale_fac has to computed to certain axis (possible values 'max','min' or int refering to dimension)
                    init_scale = -1,       # a) if init_scale = -1, already first layer consists of patches, 
                                           # b) if init_scale = sfac, it's the scale factor which the full image is scaled to
                                           # c) if init_scale = [sx,sy], it's the shape the full image is scaled to
                    keepAspect = True,     # in case of c) this keeps the apsect ratio (also if for b) if shape is not a nice number)                                     
                    smoothfac_data = 0,  # 
                    smoothfac_label = 0, #
                    interp_type = 'NN',    # nearest Neighbor (NN) or linear (lin)
                    scatter_type = 'NN',
                    normalize_input = None,
                    create_indicator_classlabels= False,
                    depth=3,               # depth of patchwork
                    ndim=2,
                    ftype=tf.float32
                    ):
    self.model = None
    self.patch_size = patch_size
    self.scale_fac = scale_fac
    self.scale_fac_ref = scale_fac_ref
    self.smoothfac_data = smoothfac_data
    self.smoothfac_label = smoothfac_label
    self.interp_type = interp_type
    self.scatter_type = scatter_type
    self.init_scale = init_scale
    self.normalize_input = normalize_input
    self.keepAspect = keepAspect
    self.create_indicator_classlabels = create_indicator_classlabels
    self.depth = depth
    self.ndim = ndim
    self.ftype=ftype


  def serialize_(self):
      return { 'patch_size':self.patch_size,
               'scale_fac' :self.scale_fac,
               'scale_fac_ref' :self.scale_fac_ref,
               'interp_type' :self.interp_type,
               'scatter_type' :self.scatter_type,
               'init_scale':self.init_scale,
               'smoothfac_data':self.smoothfac_data,
               'smoothfac_label':self.smoothfac_label,
               'normalize_input':self.normalize_input,
               'create_indicator_classlabels':self.create_indicator_classlabels,
               'keepAspect':self.keepAspect,
               'depth':self.depth,
               'ndim':self.ndim
            }


  # generates cropped data structure
  # input:
  #   trainset.shape = [batch_dim,w,h,(d),f0]
  #   labelset.shape = [batch_dim,w,h,(d),f1]
  #  trainset and label set may also be list of tensors, which is needed
  #  when  dimensions of examples differ. But note that dimensions are 
  #  only allowed to differ, if init_scale=-1 .
  # output:
  #   a list of levels (see createCropsLocal for content)
  def sample(self,trainset,labelset,
             resolutions=None,             
             generate_type='random',  # 'random' or 'tree' or 'tree_full'
             num_patches=1,           #  if 'random' this gives the number of initial draws, otherwise no function
             branch_factor=1,         #  if 'random' this gives the number of children for each random patch
             jitter=0,                #  if 'tree' this is the amount of random jitter
             jitter_border_fix=False,
             patch_size_factor=1,
             dphi=0,
             overlap=0,
             balance=None,
             augment=None,
             test=False,
             lazyEval=None,
             verbose=False):

            
    def extend_classlabels(x,class_labels_):
      if self.create_indicator_classlabels and x['labels_cropped'] is not None:
          tmp = tf.math.reduce_mean(x['labels_cropped'],list(range(1,self.ndim+1)))
          tmp = tf.cast(tmp>0.5,dtype=self.ftype)
          return tmp
      else:
          return class_labels_


    reptree = False
    if generate_type == 'tree_full':
      generate_type = 'tree'
      reptree = True


    if not isinstance(trainset,list):
      trainset = [trainset]
      labelset = [labelset]
      resolutions = [resolutions]

    N = len(trainset)

    pool = []

    for j in range(N):
        
        
      # grep and prep the labels
      labels_ = None
      class_labels_ = None
      if labelset is not None and labelset[j] is not None:
          if (self.model.classifier_train and not self.create_indicator_classlabels) and self.model.spatial_train:
              class_labels_ = labelset[j][0]
              labels_ = labelset[j][1]
          elif self.model.spatial_train and not self.model.spatial_max_train:
              labels_ = labelset[j]
          else:
              class_labels_ = labelset[j]

      # get the data 
      trainset_ = trainset[j]
      
      if self.normalize_input == 'max':
         trainset_ = trainset_/tf.reduce_max(trainset_,axis=range(1,len(trainset_.shape)))   
      if self.normalize_input == 'mean':
         trainset_ = trainset_/tf.reduce_mean(trainset_,axis=range(1,len(trainset_.shape)))   
      
      # if resolutions are also passed
      resolution_ = None
      if resolutions is not None:
          resolution_ = resolutions[j]
      
      # if we use data augmentation during training
      if augment is not None:
          print("augmenting ...")
          trainset_,labels_ = augment(trainset_,labels_)
          print("augmenting done. ")


      localCrop = lambda x,level : self.createCropsLocal(trainset_,labels_,x,
                                                         level,
                                                         generate_type,test,
                                                           num_patches=num_patches,branch_factor=branch_factor,
                                                           jitter=jitter,jitter_border_fix=jitter_border_fix,
                                                           patch_size_factor=patch_size_factor,dphi=dphi,
                                                           overlap=overlap,resolution=resolution_,balance=balance,verbose=verbose)

      # for lazy  prediction return just the function      
      if lazyEval is not None:
          return CropInstanceLazy(localCrop,self)      
          
      
      
      # do the crop in the initial level
      x = localCrop(None,0)
      x['class_labels'] = extend_classlabels(x,class_labels_)
        
      scales = [x]
      for k in range(self.depth-1):
        x = localCrop(x,k+1)        
        x['class_labels'] = extend_classlabels(x,class_labels_)    
        scales.append(x)

      # print balance info
      for k in range(self.depth):
          if scales[k]['labels_cropped'] is not None:
              labs = scales[k]['labels_cropped']
              label_range = None
              if balance is not None and 'label_range' in balance:
                    label_range = tf.cast(balance['label_range'],dtype=tf.int32)
                    labs = tf.gather(labs,label_range,axis=self.ndim+1)                            
              indicator = tf.math.reduce_max(labs,list(range(1,self.ndim+1)))
              indicator = tf.cast(indicator>0,dtype=tf.float32)    
              cur_ratio = tf.math.reduce_mean(indicator,axis=0)        
              print(' level: ' + str(k) + ' balance: ' + str(cur_ratio.numpy()) )

        
      # if we want to train in tree mode we have to complete the tree
      if reptree:
        scales = self.tree_complete(scales)

      # for repmatting the classlabels
      if scales[k]['class_labels'] is not None and (self.model.classifier_train or self.model.spatial_max_train):
          for k in range(len(scales)):
              m = scales[k]['data_cropped'].shape[0] // scales[k]['class_labels'].shape[0]              
              tmp = scales[k]['class_labels']
              if len(tmp.shape) == 1:
                  tmp = tf.expand_dims(tmp,1)
              tmp = tf.reshape(tf.tile(tf.expand_dims(tmp,1),[1,m,1]),[m*tmp.shape[0],tmp.shape[1]])              
              scales[k]['class_labels'] = tmp

        

      if len(pool) == 0:
        pool = scales
      else:
        for k in range(self.depth):
            p = pool[k]
            s = scales[k]
            # only cat those which are necessary during training
            p['data_cropped'] = tf.concat([p['data_cropped'],s['data_cropped']],0)
            if p['labels_cropped'] is not None:
              p['labels_cropped'] = tf.concat([p['labels_cropped'],s['labels_cropped']],0)            
            p['local_box_index'] = tf.concat([p['local_box_index'],s['local_box_index']],0)
            if p['class_labels'] is not None:
              p['class_labels'] = tf.concat([p['class_labels'],s['class_labels']],0)            

    intermediate_loss = True     # whether we save output of intermediate layers
    if self.model is not None:
      intermediate_loss = self.model.intermediate_loss

    return CropInstance(pool,self,intermediate_loss)



  def tree_complete(self,scales):
    b = scales[-1]['data_cropped'].shape[0]
    keys = ['data_cropped','labels_cropped','local_box_index']
    for k in range(len(scales)-1):
      s = scales[k]
      for j in keys:
        if s[j] is not None:
          multiples = [1] * (len(s[j].shape))
          multiples[0] = b//s[j].shape[0]
          s[j] = tf.tile(s[j],multiples)          
    return scales
 
  # Computes a crop of the input image stack <data_parent> according to indices 
  # given in <parent_box_index>. 
  # input - data_parent of shape [batch_dim0,w0,h0,(d1),f]
  #         parent_box_index, a index computed by convert_to_gatherND_index
  #           where parent_box_index.shape = [batch_dim1,w1,h1,(d1),nD]
  #         smoothfac - a smoothing factor (sigma of Gaussian) in pixels/voxels units
  #  batch_dim1/batch_dim0 has to be an integer such that the input can repeatedly cropped
  #  in blocks of size batch_dim0
  # output - a stack of shape [batch_dim1,w1,h1,(d1),nD]
  def crop(self,data_parent,parent_box_index,resolution,smoothfac=None,interp_type='NN',verbose=False):
      repfac = parent_box_index.shape[0] // data_parent.shape[0]
      resolution = np.array(resolution)
      res_data = []
      if smoothfac is not None:
        if (isinstance(smoothfac,float) or isinstance(smoothfac,int)) and smoothfac > 0.0:              
            if self.ndim == 2:
                  conv_gauss = conv_gauss2D_fft
            elif self.ndim == 3:
                  conv_gauss = conv_gauss3D_fft          
            sigmas = smoothfac*resolution
            if verbose:
                print(' Gaussian smooth: ', sigmas)
            data_smoothed =  conv_gauss(data_parent,tf.constant(sigmas,dtype=self.ftype))
        elif smoothfac == 'boxcar' or smoothfac == 'max' or smoothfac == 'mixture':
            if smoothfac == 'boxcar':
                if self.ndim == 2:
                      conv_box = conv_boxcar2D
                elif self.ndim == 3:
                      conv_box = conv_boxcar3D                            
            if smoothfac == 'mixture':
                if self.ndim == 2:
                      conv_box = mixture_boxcar2D
                elif self.ndim == 3:
                      conv_box = mixture_boxcar3D                            
            if smoothfac == 'max':
                if self.ndim == 2:
                      conv_box = poolmax_boxcar2D
                elif self.ndim == 3:
                      conv_box = poolmax_boxcar3D                            
            sz = np.round(resolution)
            sz[sz<1] = 1
            if np.max(sz) > 1:        
                if verbose:
                    print(' Box smooth ('+smoothfac+'): ', sz)
                data_smoothed =  conv_box(data_parent,sz)
            else:
                data_smoothed =  data_parent
        else:
            data_smoothed =  data_parent
            
      else:
        data_smoothed =  data_parent
      ds0 = data_smoothed.shape[0]
      for k in range(repfac):
          
          if interp_type == 'NN':
              tmp = tf.gather_nd(data_smoothed, parent_box_index[ds0*k:ds0*(k+1),...],batch_dims=1) 
          else:              
              tmp = self.lin_interp(data_smoothed, parent_box_index[ds0*k:ds0*(k+1),...])
          res_data.append(tmp)
              
      res_data = tf.concat(res_data,0)
      return res_data

  def lin_interp(self,data,x):

      def frac(a):
          return a-tf.floor(a)
      if self.ndim == 3:
          w = [frac(x[:,0:1,0:1,0:1, 0:1]),frac(x[:,0:1,0:1,0:1, 1:2]), frac(x[:,0:1,0:1,0:1, 2:3])]
      else:
          w = [frac(x[:,0:1,0:1, 0:1]),frac(x[:,0:1,0:1, 1:2]) ]

      def gather(d,idx,s):
          q = tf.convert_to_tensor(s)
          for k in range(self.ndim+1):
              q = tf.expand_dims(q,0)
          idx = idx + q
          weight = 1.0
          for k in range(self.ndim):
              if s[k] == 1:
                  weight = weight*w[k]
              else:
                  weight = weight*(1-w[k])
                  
          return tf.gather_nd(d, idx ,batch_dims=1) * weight
      x = tf.cast(x,dtype=tf.int32)
      if self.ndim == 3:
          res = gather(data,x,[0,0,0]) + gather(data,x,[1,0,0]) + gather(data,x,[0,1,0]) + gather(data,x,[0,0,1]) + gather(data,x,[0,1,1]) + gather(data,x,[1,0,1]) + gather(data,x,[1,1,0]) + gather(data,x,[1,1,1])
      else:
          res = gather(data,x,[0,0]) + gather(data,x,[1,0]) + gather(data,x,[0,1]) + gather(data,x,[1,1]) 
      return res
      
      


  # Converts normalized coordinates (as used in tf.crop_and_resize) in local_boxes 
  # to actual pixel coordinates, it assumed that all boxes are of the same size!! 
  #  local_boxes - normalized coordinates of shape [N,4] (2D) or [N,6] (3D), 
  #  sz - abs. size of images from which crop is performed (list of size nD)
  #  patch_size - size of patches (list of size nD) 
  def convert_to_gatherND_index(self,local_boxes,sz,patch_size,interp_type='NN',aspects=None):
      nD = self.ndim

      if sz is None:             
         sz = [None] * (nD+1)
         for k in range(nD):
           sz[k+1] = tf.math.round(patch_size[k]/(local_boxes[0,k+nD]-local_boxes[0,k]) )

      rans = [None] * nD
      center = [None] * nD
      for k in range(nD):
        wid = (local_boxes[0,(k+nD):(k+nD+1)]-local_boxes[0,k:k+1])
        scfac = wid/patch_size[k]*sz[k+1]
        rans[k] = tf.cast(tf.range(patch_size[k]),dtype=self.ftype)*scfac-patch_size[k]*0.5*scfac
        center[k] = (local_boxes[:,k:(k+1)] + local_boxes[:,(k+nD):(k+nD+1)])*0.5 * sz[k+1]

      center = tf.transpose(center,[1,0,2]);
      res_shape = [1] * (nD+2)
      res_shape[0] = local_boxes.shape[0]
      res_shape[nD+1] = nD
      center = tf.reshape(center,res_shape)
      qwq = tf.expand_dims(rep_rans(rans,patch_size,nD),0)


      if local_boxes.shape[1] > nD*2:
            qwq = tf.tile(qwq,[local_boxes.shape[0]]+[1]*(nD+1))
            phi = local_boxes[:,(2*nD):]
            if nD == 2:
                phi = tf.expand_dims(local_boxes[:,(2*nD):],2)
                a = tf.math.cos(phi)
                b = tf.math.sin(phi)
                R = tf.concat([tf.concat([a,b],1),
                               tf.concat([-b,a],1)],2)
            else:
                R = quaternion2mat(phi)  
                c=lambda x: tf.expand_dims(tf.concat(x,1),2)
                R = tf.concat([c(R[0]),c(R[1]),c(R[2])],2)
    
            if aspects is not None:
                aspects = tf.cast(aspects,dtype=self.ftype)
                R = tf.einsum('bij,i,j->bij',R,aspects,1/aspects)
                #f = aspects
                #p = f[1]/f[0]
                #R = tf.concat([tf.concat([a,b*p],1),
                #               tf.concat([-b/p,a],1)],2)
            if nD==2:
                qwq = tf.einsum('bxyi,bij->bxyj',qwq,R)
            else:
                qwq = tf.einsum('bxyzi,bij->bxyzj',qwq,R)


      local_box_index = center+qwq

      if interp_type == 'NN': # cast index to int
          local_box_index = tf.dtypes.cast(tf.floor(local_box_index+0.5),dtype=tf.int32)
    
          ## clip indices
      lind = []
      for k in range(nD):
          tmp = local_box_index[...,k:(k+1)]
          tmp = tf.math.maximum(tmp,0)
          if interp_type == 'NN': 
             tmp = tf.math.minimum(tmp,tf.cast(sz[k+1]-1,tf.int32))
          else:
             tmp = tf.math.minimum(tmp,sz[k+1]-2)
          lind.append(tmp)
      local_box_index = tf.concat(lind,nD+1)

    



      return local_box_index, sz;


  # Computes normalized coordinates of random crops
  #  bbox_sz - a list of size 4 (2D) or 6 (3D) representing the template of the box
  #              with origin zero.
  #  numboxes - number of boxes to be distributed
  # returns boxes of shape [numboxes,4] (2D) or [numboxes,6] (3D) suitable as
  #  input for convert_to_gatherND_index
  def random_boxes(self,bbox_sz,labels, numboxes,balance,dphi=0):
      
          
      
      
      def draw_rnd(label,M):        

        ratio = balance['ratio']
        N = 10000
        numrounds = 1000
        if 'N' in balance:
            N = balance['N']
        if 'numrounds' in balance:
            numrounds = balance['numrounds']
        label_range = None
        label_weight = None
        if 'label_range' in balance:
            label_range = tf.cast(balance['label_range'],dtype=tf.int32)
        if 'label_weight' in balance:
            label_weight = tf.cast(balance['label_weight'],dtype=self.ftype)
            for k in range(nD):
                label_weight = tf.expand_dims(label_weight,0)
          
        points_tot = []
        for k in range(label.shape[0]):
            L = label[k,...]
            if label_range is not None:
                L = tf.gather(L,label_range,axis=nD)
            if label_weight is not None:
                L = L*label_weight
            L = np.amax(L,nD)
            L = 1.0*(L>0)
            sz = L.shape
            cnt=0

            numvx = np.prod(sz)
            pos = tf.reduce_sum(L)
            neg = numvx-pos
            
            if  pos == 0 or pos-numvx*ratio > -0.01:
                print("warning: cannot achieve desired sample ratio, taking uniform")
                P = L*0+1
            else:
                background_p = (1-ratio)*pos/(numvx*ratio-pos)
                P = (L + background_p)
                
            p = tf.reshape(P,[numvx])
            p = tf.cast(p,dtype=tf.float64)
                
            p = p/tf.reduce_sum(p)
            idx = np.random.choice(numvx,M,p=p)
            R = np.transpose(np.unravel_index(idx,sz))
            R = R + np.random.uniform(low=0,high=1,size=R.shape)
            points = R/sz
      


            # points = []
            # if N < M:
            #     N = M # num drawn samples per round
            # for j in range(numrounds):
            #     R = np.random.uniform(low=0,high=1,size=(N,nD))
            #     for i in range(nD):            
            #         R[:,i] = np.floor(R[:,i]*sz[i])
            #     Q = tf.gather_nd(L,tf.convert_to_tensor(R,dtype=tf.int32))
            #     pos = tf.reduce_sum(Q)
            #     neg = N-pos
            #     if  pos-N*ratio > -0.01:
            #         print("warning: cannot achieve desired sample ratio, taking uniform")
            #         P = Q*0+1
            #     else:
            #         background_p = (1-ratio)*pos/(N*ratio-pos)
            #         P = (Q + background_p)
            #         P = P / tf.reduce_max(P)


            #     idx = np.argwhere( (P > np.random.uniform(low=0,high=1,size=(N))))
            #     R = tf.gather(R,idx[:,0],axis=0)
            #     R = R + np.random.uniform(low=0,high=1,size=R.shape)
            #     R = R/sz
            #     points.append(R)
            #     cnt = cnt + R.shape[0]
            #     if cnt >= M:
            #         break
            # if cnt < M:
            #     print("warning: cannot achieve desired sample ratio, taking uniform (2)")               
            #     return None
                
            # points = tf.concat(points,0)
            # points = points[0:M,:]
            
            
            
            points_tot.append(points)
        
        centers = tf.concat(points_tot,0)            
        centers = tf.tile(centers,[1,2]).numpy()
            
        return centers

      def draw_uniform():
          centers = [None]*(nD*2)
          for k in range(nD):
            c = np.random.uniform(0,1,(numboxes, 1))
            centers[k] = c
            centers[k+nD] = c
          centers = np.concatenate(centers,1)
          return centers
        

      nD = self.ndim
      
      if labels is None or balance is None:
          centers = draw_uniform()
      else:
          centers = draw_rnd(labels,M=numboxes//labels.shape[0])
          if centers is None:
              centers = draw_uniform()
              
      local_boxes = centers + bbox_sz
      
      if dphi > 0:
          if nD==2:
              local_boxes = np.concatenate([local_boxes,np.random.uniform(-dphi,dphi,[local_boxes.shape[0],1])],1)
          else:
              quats = np.random.normal(0,dphi,[local_boxes.shape[0],3])
              local_boxes = np.concatenate([local_boxes,quats],1)
              
      
      for k in range(nD):
            idx = local_boxes[:,k]<0
            local_boxes[idx,nD+k] = local_boxes[idx,nD+k] - local_boxes[idx,k]
            local_boxes[idx,k] = 0
            idx = local_boxes[:,k+nD]>1
            local_boxes[idx,k] = local_boxes[idx,k] - (local_boxes[idx,k+nD]-1)
            local_boxes[idx,nD+k] = 1
      local_boxes = tf.convert_to_tensor(local_boxes,dtype=self.ftype)
            
      return local_boxes
  
    

    
  

  # Computes normalized coordinates of random crops
  #  bbox_sz - a list of size 4 (2D) or 6 (3D) representing the template of the box
  #              with origin zero.
  #  overlap - an integer giving the additoinal number of boxes per dimension
  #  jitter - to add random jitter 
  # return boxes of shape [numboxes,4] (2D) or [numboxes,6] (3D) 
  def tree_boxes(self,bbox_sz,overlap,jitter=0,jitter_border_fix=False,dphi=0):

      nD = self.ndim
      centers = [None] * nD
      nums = [None] * nD
      totnum = 1
      for k in range(nD):
        delta = bbox_sz[nD+k]-bbox_sz[k]
        nums[k] = np.floor(1/delta)+1+overlap
        if nums[k] < 2:
            nums[k] = 2
        delta_small= (1-delta)/(nums[k]-1) -0.000001
        frac = nums[k]*delta-1
        centers[k] = tf.cast(tf.range(nums[k]),dtype=self.ftype)*delta_small + delta*0.5
        totnum *= nums[k]
      centers = rep_rans(centers,nums,nD)
      
      if jitter > 0:
          sh = centers.shape
          rands = []
          for k in range(nD):
             delta = bbox_sz[nD+k]-bbox_sz[k]              
             rng = np.random.uniform(-delta/2*jitter,delta/2*jitter,sh[0:nD])
             if jitter_border_fix:
                if nD == 2:
                    if k == 0: 
                        rng[0,:] = 0
                        rng[-1,:] = 0
                    elif k == 1:
                        rng[:,0] = 0
                        rng[:,-1] = 0
                else:
                     if k == 0: 
                         rng[0,:,:] = 0
                         rng[-1,:,:] = 0
                     elif k == 1:
                         rng[:,0,:] = 0
                         rng[:,-1,:] = 0
                     else:
                         rng[:,:,0] = 0
                         rng[:,:,-1] = 0
             else:
                 if nD == 2:
                     if k == 0: 
                         rng[0,:] = tf.math.abs(rng[0,:])
                         rng[-1,:] = -tf.math.abs(rng[-1,:])
                     elif k == 1:
                         rng[:,0] = tf.math.abs(rng[:,0])
                         rng[:,-1] = -tf.math.abs(rng[:,-1])
                 else:
                      if k == 0: 
                         rng[0,:,:] = tf.math.abs(rng[0,:,:])
                         rng[-1,:,:] = -tf.math.abs(rng[-1,:,:])
                      elif k == 1:
                         rng[:,0,:] = tf.math.abs(rng[:,0,:])
                         rng[:,-1,:] = -tf.math.abs(rng[:,-1,:])
                      else:
                         rng[:,:,0] = tf.math.abs(rng[:,:,0])
                         rng[:,:,-1] = -tf.math.abs(rng[:,:,-1])
             rands.append(np.expand_dims(rng,nD))
          rands = np.concatenate(rands,nD)
          centers = centers + rands

      qwq = tf.tile(tf.reshape(centers,tf.cast([totnum,nD],dtype=tf.int32)),[1,2]) + bbox_sz
      
      if dphi > 0:
          qwq = np.concatenate([qwq,np.random.uniform(-dphi,dphi,[qwq.shape[0],1])],1)
      qwq = tf.cast(qwq,dtype=self.ftype)
      
      
      return qwq, qwq.shape[0];


  def createCropsLocal(self,data_parent,labels_parent,crops,
      level,
      generate_type,test,
      jitter=0,
      jitter_border_fix=False,
      num_patches=1,
      branch_factor=1,
      dphi=0,
      patch_size_factor=1,
      overlap=0,resolution=None,balance=None,verbose=True):
      
      
      
      def get_patchsize(level):   # patch_size could be eg [32,32], or a list [ [32,32], [32,32] ] corresponding to differne levels
            if isinstance(self.patch_size[0],Iterable):
                return self.patch_size[level]
            else: 
                return self.patch_size
              
    
      def get_scalefac(level,abssz):  # either a float, or a list of floats where each entry corresponds to a different depth level,
                                  # or dict with entries { 'level0' : [0.5,0.5] , 'level1' : [0.4,0.3]} where scalefac is dependent on dimension and level
          if isinstance(self.scale_fac,float) or isinstance(self.scale_fac,int):
              return [self.scale_fac]*self.ndim
          elif isinstance(self.scale_fac,dict):
              if ('level' + str(level))  not in self.scale_fac:
                  return None
              tmp = self.scale_fac['level' + str(level)]
              if isinstance(tmp,float) or isinstance(tmp,int):
                  return [tmp]*self.ndim
              elif isinstance(tmp,str):
                  absv = tmp.replace("mm","").split(",")
                  absv = list(map(int, absv))
                  tmp = []
                  for k in range(self.ndim):
                      tmp.append(absv[k]/abssz[k])
                  return tmp
              else:
                  return tmp
          elif isinstance(self.scale_fac,Iterable):
              return [self.scale_fac[level]]*self.ndim
      
      def get_smoothing(level,which):
          if which == 'data':
              sm = self.smoothfac_data
          else:
              sm = self.smoothfac_label
          if isinstance(sm,list):
              return sm[level]
          else:
              return sm
        
      last_abssz = None
      if crops is not None:
          last_abssz = crops['absolute_size_patch_mm']
      this_scale_fac=get_scalefac(level,last_abssz)
      
      
      init_scale = self.init_scale
      keepAspect = self.keepAspect
      divisor = 8 # used for initital scale to get a nice image size
      nD = self.ndim
      forwarded_aspects = np.ones(nD) #[1]*nD
      aspect_correction = np.ones(nD) #nD*[1]
      if keepAspect and crops is not None:
          aspect_correction = crops['aspect_correction']

      if crops is None:
        images = data_parent
        if generate_type == 'random':     # 
           replicate_patches = num_patches
        elif generate_type == 'tree':     # 
          replicate_patches = 1
        else:
          assert 'not valid generate_type'
      else:
        images = crops['data_cropped']
        if generate_type == 'random':     #
          replicate_patches = branch_factor
        elif generate_type == 'tree':     # 
          replicate_patches = None
        else:
          assert 'not valid generate_type'
      sz = images.shape
      bsize = sz[0] 
      
      if generate_type == 'tree' and crops is None:
        assert bsize == 1, "for generate_type=tree the batch_size has to be one!"
 
    
      
      start = timer()

      ############### compute bbox coordinates
      if crops is None and isinstance(init_scale,list):      # in case init_scale = shape, we scale the full image
        patch_size = init_scale                              # onto shape, which is then first layer
        qbox = [None] * (2*nD)
        for d in range(nD):
          qbox[d] = tf.zeros(shape=(replicate_patches*bsize, 1))
          qbox[d+nD] = tf.ones(shape=(replicate_patches*bsize, 1))
        local_boxes = tf.concat(qbox,1)
        # compute relative aspects to keep aspectration
        fac = max(sz[1:nD+1])/max(patch_size)
        for d in range(nD):
          forwarded_aspects[d] = fac * patch_size[d]/sz[d+1]
      elif crops is None and init_scale != -1 and not isinstance(init_scale,str):               # the first layer is a isotropically scaled version
   

        patch_size = [0] * nD        
        qbox = [None] * (2*nD)
        for d in range(nD):
          desired = round(init_scale*sz[d+1])
          patch_size[d] = getClosestDivisable(desired,divisor)
          fac = 1 #patch_size[d]/desired
          qbox[d] = tf.zeros(shape=(replicate_patches*bsize, 1))
          qbox[d+nD] = fac*tf.ones(shape=(replicate_patches*bsize, 1))
        local_boxes = tf.concat(qbox,1)
        # compute relative aspects to keep aspectration
        fac = max(sz[1:nD+1])/max(patch_size)
        for d in range(nD):
          forwarded_aspects[d] = fac * patch_size[d]/sz[d+1]
      else:                                  # the first layer is already patched                

        patch_size_=get_patchsize(level)
        
        patch_size = [0]*nD
        for k in range(len(patch_size)):
            patch_size[k] = patch_size_[k]*patch_size_factor

        if crops is None and isinstance(init_scale,str):
            forwarded_aspects = [1]*nD
            bbox_sz = [0] * (nD*2)            
            if init_scale.find('mm') != -1 or init_scale.find('cm') != -1:
                assert (resolution is not None), "for absolute init_scale you have to pass resolution"
                if init_scale.find('cm') != -1:
                    sizes_mm = init_scale.replace("cm","").split(",")
                    sizes_mm = list(map(lambda x: int(x)*10, sizes_mm))
                else:
                    sizes_mm = init_scale.replace("mm","").split(",")
                    sizes_mm = list(map(int, sizes_mm))
                for d in range(nD):
                    sfac = patch_size_factor*sizes_mm[d]/resolution[d]/sz[d+1]
                    bbox_sz[d]    = -sfac*0.5
                    bbox_sz[d+nD] = sfac*0.5
            if init_scale.find('vx') != -1:
                sizes_vx = init_scale.replace("vx","").split(",")
                sizes_vx = list(map(int, sizes_vx))
                for d in range(nD):
                    sfac = patch_size_factor*sizes_vx[d]/sz[d+1]
                    bbox_sz[d]    = -sfac*0.5
                    bbox_sz[d+nD] = sfac*0.5
                            
            
        else:
            
            asp = []
            for d in range(nD):
                asp.append(sz[d+1]/patch_size[d]*this_scale_fac[d]*0.5)
            if self.scale_fac_ref == 'max':
                asp = max(asp)
            elif self.scale_fac_ref == 'min':
                asp = min(asp)
            else:
                asp = asp[self.scale_fac_ref]
                
            bbox_sz = [0] * (nD*2)
            for d in range(nD):
                bbox_sz[d]   = -asp*patch_size[d]/sz[d+1]* aspect_correction[d]
                bbox_sz[d+nD] = asp*patch_size[d]/sz[d+1] *aspect_correction[d]


        if generate_type == 'random':     
          local_boxes = self.random_boxes(bbox_sz,labels_parent,bsize*replicate_patches,balance,dphi=dphi)
        elif generate_type == 'tree':   
          if crops is not None:
              local_boxes = []
              for t in range(crops['parent_boxes'].shape[0]):
                lb,replicate_patches = self.tree_boxes(bbox_sz,overlap,jitter=jitter,jitter_border_fix=jitter_border_fix,dphi=dphi)
                local_boxes.append(tf.expand_dims(lb,1))
              local_boxes = tf.concat(local_boxes,1)
          else:
              local_boxes,replicate_patches = self.tree_boxes(bbox_sz,overlap,jitter=jitter,jitter_border_fix=jitter_border_fix,dphi=dphi)
            
          



    
      ############### compute box indices
      if crops is None: # the first layer
        parent_boxes = local_boxes
      else: # preceding layers (crops have to recomputed according to grandparent)
        last_boxes = crops['parent_boxes']
          
        if generate_type == 'tree':          
           ccatdim=2
           last_boxes = tf.expand_dims(last_boxes,0)
        else:
           ccatdim=1
           last_boxes = tf.tile(last_boxes,[replicate_patches,1])                   
          
        center_last = []
        center_local = []
        wid = []
        wid_local = []            
        for j in range(nD):
            center_last.append( (last_boxes[...,(nD+j):(nD+1+j)]+last_boxes[...,j:(j+1)])*0.5 )
            center_local.append( (local_boxes[...,(nD+j):(nD+1+j)]+local_boxes[...,j:(j+1)])*0.5 - 0.5 )
            wid.append( (last_boxes[...,(nD+j):(j+nD+1)]-last_boxes[...,j:(j+1)]) )
            wid_local.append(local_boxes[...,(nD+j):(j+nD+1)]-local_boxes[...,j:(j+1)])

        if last_boxes.shape[ccatdim] > nD*2:
            phi = last_boxes[...,2*nD:]
        else:
            phi = last_boxes[...,0:(1+(nD-2)*2)]*0
        
        center_ = center_last
        if nD == 2:
            R = [ [tf.math.cos(phi),tf.math.sin(phi)],
                  [-tf.math.sin(phi),tf.math.cos(phi)]  ]
        else:
            R = quaternion2mat(phi)

        for a in range(nD):
            for b in range(nD):
                center_[a] += R[a][b]*center_local[b]*wid[a]*sz[1+b]/sz[1+a]

        toconcat = [None] * (2*nD)
        for a in range(nD):
            toconcat[a] =    center_[a]-wid[a]*wid_local[a]*0.5
            toconcat[a+nD] = center_[a]+wid[a]*wid_local[a]*0.5
        if nD==2:
            toconcat.append(phi+local_boxes[...,2*nD:])
        else:
            toconcat.append(quaternion_prod(phi,local_boxes[...,2*nD:]))
        parent_boxes = tf.concat(toconcat,ccatdim)

        if generate_type == 'tree':                  
            parent_boxes = tf.reshape(parent_boxes,[parent_boxes.shape[0]*parent_boxes.shape[1], parent_boxes.shape[2]])
            local_boxes = tf.reshape(local_boxes,[local_boxes.shape[0]*local_boxes.shape[1],  local_boxes.shape[2]])

            
  #          p = sz[2]/sz[1]
                                
  #          centerX_ = center_last[0] + (tf.math.cos(phi)*center[0]_local*wid[0]*sz[1]/sz[1] + tf.math.sin(phi)*center[1]_local*wid[0]*sz[2]/sz[1])
  #          centerY_ = center_last[1] + (-tf.math.sin(phi)*center[0]_local*wid[1]*sz[1]/sz[2] + tf.math.cos(phi)*center[1]_local*wid[1]*sz[2]/sz[2])

            # centerX_last = (last_boxes[...,(nD):(nD+1)]+last_boxes[...,0:(0+1)])*0.5
            # centerY_last = (last_boxes[...,(1+nD):(1+nD+1)]+last_boxes[...,1:(2)])*0.5
            # centerX_local = (local_boxes[...,(nD):(nD+1)]+local_boxes[...,0:(0+1)])*0.5 - 0.5
            # centerY_local = (local_boxes[...,(1+nD):(1+nD+1)]+local_boxes[...,1:(2)])*0.5 - 0.5
            # widX =  (last_boxes[...,(nD):(nD+1)]-last_boxes[...,0:(0+1)])
            # widY = (last_boxes[...,(1+nD):(1+nD+1)]-last_boxes[...,1:(2)])
            # widX_local = (local_boxes[...,(nD):(nD+1)]-local_boxes[...,0:(0+1)])
            # widY_local = (local_boxes[...,(1+nD):(1+nD+1)]-local_boxes[...,1:(2)])

            # centerX_ = centerX_last + (tf.math.cos(phi)*centerX_local*widX + tf.math.sin(phi)*centerY_local*widX*p)
            # centerY_ = centerY_last + (-tf.math.sin(phi)*centerX_local*widY/p + tf.math.cos(phi)*centerY_local*widY)
  
            # print("###################################")
            # print("forw_asp", forwarded_aspects)
            # print("patch_size", patch_size_)
            # print("sz", sz)                      
            # print("###################################")
    
            #toconcat = [centerX_-widX*widX_local*0.5,centerY_-widY*widY_local*0.5, 
            #            centerX_+widX*widX_local*0.5,centerY_+widY*widY_local*0.5,phi+local_boxes[...,2*nD:]]
            #parent_boxes = tf.concat(toconcat,ccatdim)
            
            
      

      dest_full_size = [None]*(nD+1)
      for k in range(nD):
          dest_full_size[k+1] = tf.convert_to_tensor(np.math.floor(1/patch_size_factor*patch_size[k]/np.min(parent_boxes[:,nD+k]-parent_boxes[:,k])),dtype=self.ftype)

      # compute the index suitable for gather_nd
      local_box_index,_ = self.convert_to_gatherND_index(local_boxes,sz,patch_size,interp_type=self.interp_type,
                                                         aspects=None)
      parent_box_index,_ = self.convert_to_gatherND_index(parent_boxes,data_parent.shape,patch_size,interp_type=self.interp_type,
                                                          aspects=aspect_correction)
      parent_box_scatter_index, _ = self.convert_to_gatherND_index(parent_boxes,dest_full_size,patch_size,interp_type=self.scatter_type,
                                                           aspects=aspect_correction)
  


      relres = []
      
      for k in range(nD):
        relres.append(((parent_boxes[0,k+nD]-parent_boxes[0,k])*data_parent.shape[k+1]/patch_size[k]).numpy() ) 
                    
      abssz = None
      if resolution is not None:
          abssz = resolution[0:nD] * np.array(relres) * patch_size[0:nD]
  
      if verbose:
        print("--------- cropping, level ",level)
        print("shape of patch: ", *patch_size )
        print("voxsize (relative to original scale): ", *relres)
        if resolution is not None:
            print("patchsize (mm): ", *abssz)
        print("numpatches in level: %d" % (parent_box_index.shape[0] / data_parent.shape[0]))
        print("shape of full output: ",  *list(map(lambda x: x.numpy(), dest_full_size[1:])))
        



      ############## do the actual cropping
      res_data = self.crop(data_parent,parent_box_index,relres,get_smoothing(level,'data'),interp_type=self.interp_type,verbose=verbose)        
      if labels_parent is not None:      
        res_labels = self.crop(labels_parent,parent_box_index,relres,get_smoothing(level,'label'),interp_type=self.interp_type,verbose=verbose)
      else:
        res_labels = None

  #    if res_data.shape[0] != local_boxes.shape[0]:
  #        res_data


      # for testing
      if test:
        if crops is None:
          mult = (nD+2)*[1]
          mult[0] = num_patches
          images = tf.tile(images,mult)
        test = tf.gather_nd(images,local_box_index,batch_dims=1)
      else:
        test = None

  
      if verbose:
        end = timer()
        print("time elapsed: " + str(end - start) )


      return {"data_cropped" : res_data, 
              "labels_cropped" : res_labels, 
              
               # only used for testing (crops of the last scale)
              "data_cropped_test": test,
              
              # these are crop coordinates refering to the last upper scale
              "local_boxes" : local_boxes, 
              "local_box_index": local_box_index,

              # these are crop coordinates refereing to the very original image
              "parent_boxes": parent_boxes, 
              "parent_box_index": parent_box_index,
              "parent_box_scatter_index": parent_box_scatter_index,
              "dest_full_size":dest_full_size,

              "absolute_size_patch_mm":abssz,
              "aspect_correction":forwarded_aspects
              }



  def take_subselection(self, x, idx):
        if idx is None:
            return x
        props = ["data_cropped" , 
                 'labels_cropped',
                 "local_boxes",
                 'local_box_index',
                 'parent_boxes',
                 'parent_box_index',
                 'parent_box_scatter_index',
                 'class_labels']
        for p in props:
            if p in x:
                if x[p] is not None:
                    x[p] = tf.gather(x[p],(idx),axis=0)      
        return x
                

  ############## teststuff

  def testtree(self,im):
    print(im.shape)
    c = self.sample(im[0:1,...],None,test=False,generate_type='tree',jitter=0.1,verbose=True)
    self.showtest(c)

  def testrandom(self,im):
    c = self.sample(im[0:1,...],None,test=False,generate_type='random',verbose=True,num_patches=300)
    self.showtest(c)

  def scatter_valid(self, index, data, size):
      return tf.scatter_nd(index,data,size) 

  def showtest(self,c):
    
    import matplotlib.pyplot as plt
      
    data_ = c.getInputData()
    f = plt.figure(figsize=(10,10))
    for level in range(self.depth):

      dqq = data_['input'+str(level)]
      print(dqq.shape)
      pbox_index = c.scales[level]['parent_box_scatter_index']
      sha = c.scales[level]['dest_full_size']

      ax = plt.subplot(2,self.depth,level+1)
      qq = self.scatter_valid(pbox_index,dqq,[sha[1],sha[2],1]) /  self.scatter_valid(pbox_index,dqq*0+1,[sha[1],sha[2],1])
      ax.imshow(tf.transpose(qq[:,:,0],[0,1]))

      ax = plt.subplot(2,self.depth,level+1+self.depth)
      qq = self.scatter_valid(pbox_index,dqq*0+1,[sha[1],sha[2],1])
      ax.imshow(tf.transpose(qq[:,:,0],[0,1]))




def quaternion2mat(q):
    x = q[...,0:1]
    y = q[...,1:2]
    z = q[...,2:3]
                   
    w = tf.math.sqrt(tf.maximum(0.0,1.0-(x*x+y*y+z*z)));
    Rxx = 1 - 2*(y*y + z*z);
    Rxy = 2*(x*y - z*w);
    Rxz = 2*(x*z + y*w);
    Ryx = 2*(x*y + z*w);
    Ryy = 1 - 2*(x*x + z*z);
    Ryz = 2*(y*z - x*w );
    Rzx = 2*(x*z - y*w );
    Rzy = 2*(y*z + x*w );
    Rzz = 1 - 2 *(x*x + y*y);
    return [ [Rxx,Rxy,Rxz],
             [Ryx,Ryy,Ryz], 
             [Rzx,Rzy,Rzz] ]

def quaternion_prod(x,y):

    p1 = x[...,0:1]
    p2 = x[...,1:2]
    p3 = x[...,2:3]
    
    q1 = y[...,0:1]
    q2 = y[...,1:2]
    q3 = y[...,2:3]

    p0 = tf.math.sqrt(tf.maximum(0.0,1.0-(p1*p1+p2*p2+p3*p3)));
    q0 = tf.math.sqrt(tf.maximum(0.0,1.0-(q1*q1+q2*q2+q3*q3)));
  
    
    #real = p0*q0 − (p1*q1 + p2*q2 + p3*q3),
    return tf.concat([(p2*q3 - p3*q2) + (p2*q3 - p3*q2) + p0*q1 + q0*p1,
                      (p3*q1 - p1*q3) + (p3*q1 - p1*q3) + p0*q2 + q0*p2,
                      (p1*q2 - p2*q1) + (p1*q2 - p2*q1) + p0*q3 + q0*p3],-1)







#if 0:

 # crops = c.sample(trainset,None,test=False,generate_type='tree',num_patches = 1)

  #  # plt.imshow(tf.squeeze(trainset[0,:,:,0]))

  #   f = plt.figure(figsize=(20,20))

  #   cnt = 1
  #   n = 0
  #   for x in scales:
  # # 
  #     print(x['data_cropped'].shape)
  #     ax = plt.subplot(4,3,cnt)
  #     ax.imshow(tf.squeeze(x['data_cropped'][n,:,:,0]))
  #     cnt = cnt + 1

  #     ax = plt.subplot(4,3,cnt)
  #     ax.imshow(tf.squeeze(x['data_cropped_test'][n,:,:,0]))
  #     cnt = cnt + 1

  #     ax = plt.subplot(4,3,cnt)
  #     ax.imshow(tf.math.reduce_sum(x['labels_cropped'][n,:,:,:],2))
  #     cnt = cnt + 1



# cgen = CropGenerator(patch_size = (64,64), 
#                   scale_fac = 0.6, 
#                   init_scale = -1,
#                   overlap = 10,
#                   depth=3)
#cgen.testtree(labelset[0][0:1,:,:,5:6])








