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

  # get input training data 
  def getInputData(self):
            
    cnt = 0
    inp = {}
    for x in self.scales:
      inp['input'+str(cnt)] = x['data_cropped']
      inp['cropcoords'+str(cnt)] = x['local_box_index']
      cnt = cnt + 1
    return inp

  # get target training data 
  def getTargetData(self):
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
        
    if classifier_train and not create_indicator_classlabels:
        out.append(self.scales[-1]['class_labels'])
    if spatial_train:
        out.append(self.scales[-1]['labels_cropped'])
    
    
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
             overlap=0,
             balance=None,
             augment=None,
             test=False,
             lazyEval=None,
             verbose=False):

    def get_patchsize(level):   # patch_size could be eg [32,32], or a list [ [32,32], [32,32] ] corresponding to differne levels
        if isinstance(self.patch_size[0],Iterable):
            return self.patch_size[level]
        else: 
            return self.patch_size
          

    def get_scalefac(level):  # either a float, or a list of floats where each entry corresponds to a different depth level,
                              # or dict with entries { 'level0' : [0.5,0.5] , 'level1' : [0.4,0.3]} where scalefac is dependent on dimension and level
      if isinstance(self.scale_fac,float):
          return [self.scale_fac]*self.ndim
      elif isinstance(self.scale_fac,dict):
          tmp = self.scale_fac['level' + str(level)]
          if isinstance(tmp,float):
              return [tmp]*self.ndim
          else:
              return tmp
      elif isinstance(self.scale_fac,Iterable):
          return [self.scale_fac[level]]*self.ndim
            
    def extend_classlabels(x,class_labels_):
      if self.create_indicator_classlabels and x['labels_cropped'] is not None:
          return tf.expand_dims(tf.math.reduce_max(x['labels_cropped'],list(range(1,self.ndim+2))),1)
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
      if labelset[j] is not None:
          if (self.model.classifier_train and not self.create_indicator_classlabels) and self.model.spatial_train:
              class_labels_ = labelset[j][0]
              labels_ = labelset[j][1]
          elif self.model.spatial_train:
              labels_ = labelset[j]
          else:
              class_labels_ = labelset[j]

      # get the data 
      trainset_ = trainset[j]
      
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
                                                         get_patchsize(level), get_scalefac(level),
                                                         generate_type,test,
                                                           num_patches=num_patches,branch_factor=branch_factor,
                                                           jitter=jitter,jitter_border_fix=jitter_border_fix,
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
              indicator = tf.cast(indicator>0.5,dtype=tf.float32)    
              cur_ratio = tf.math.reduce_mean(indicator,axis=0)        
              print(' level: ' + str(k) + ' balance: ' + str(cur_ratio.numpy()) )

        
      # if we want to train in tree mode we have to complete the tree
      if reptree:
        scales = self.tree_complete(scales)

      # for repmatting the classlabels
      if scales[k]['class_labels'] is not None and self.model.classifier_train:
          for k in range(len(scales)):
              m = scales[k]['data_cropped'].shape[0] // scales[k]['class_labels'].shape[0]              
              tmp = scales[k]['class_labels']
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
  def crop(self,data_parent,parent_box_index,smoothfac=None,interp_type='NN'):
      repfac = parent_box_index.shape[0] // data_parent.shape[0]
      res_data = []
      if self.ndim == 2:
          conv_gauss = conv_gauss2D_fft
      elif self.ndim == 3:
          conv_gauss = conv_gauss3D_fft
          
      if smoothfac is not None and smoothfac > 0.0:
        data_smoothed =  conv_gauss(data_parent,tf.constant(smoothfac,dtype=self.ftype))
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
  def convert_to_gatherND_index(self,local_boxes,sz,patch_size,interp_type='NN'):
      nD = self.ndim

      if sz is None:             
         sz = [None] * (nD+1)
         for k in range(nD):
           sz[k+1] = tf.math.round(patch_size[k]/(local_boxes[0,k+nD]-local_boxes[0,k]) )

      rans = [None] * nD
      start_abs = [None] * nD
      for k in range(nD):
        scfac = (local_boxes[0,(k+nD):(k+nD+1)]-local_boxes[0,k:k+1])/patch_size[k]*sz[k+1]
        rans[k] = tf.cast(tf.range(patch_size[k]),dtype=self.ftype)*scfac
        start_abs[k] = local_boxes[:,k:(k+1)] * sz[k+1]

      start_abs = tf.transpose(start_abs,[1,0,2]);
      res_shape = [1] * (nD+2)
      res_shape[0] = local_boxes.shape[0]
      res_shape[nD+1] = nD
      start_abs = tf.reshape(start_abs,res_shape)

      qwq = tf.expand_dims(rep_rans(rans,patch_size,nD),0)

      local_box_index = start_abs+qwq

      if interp_type == 'NN': # cast index to int
          local_box_index = tf.dtypes.cast(tf.floor(local_box_index+0.5),dtype=tf.int32)
    
          ## clip indices
          lind = []
          for k in range(nD):
            tmp = local_box_index[...,k:(k+1)]
            tmp = tf.math.maximum(tmp,0)
            tmp = tf.math.minimum(tmp,tf.cast(sz[k+1]-1,tf.int32))
            lind.append(tmp)
          local_box_index = tf.concat(lind,nD+1)

    



      return local_box_index, sz;


  # Computes normalized coordinates of random crops
  #  bbox_sz - a list of size 4 (2D) or 6 (3D) representing the template of the box
  #              with origin zero.
  #  numboxes - number of boxes to be distributed
  # returns boxes of shape [numboxes,4] (2D) or [numboxes,6] (3D) suitable as
  #  input for convert_to_gatherND_index
  def random_boxes(self,bbox_sz,labels, numboxes,balance):
      
          
      
      
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
            sz = L.shape
            cnt=0
            points = []
            if N < M:
                N = M # num drawn samples per round
            for j in range(numrounds):
                R = np.random.uniform(low=0,high=1,size=(N,nD))
                for i in range(nD):            
                    R[:,i] = np.floor(R[:,i]*sz[i])
                Q = tf.gather_nd(L,tf.convert_to_tensor(R,dtype=tf.int32))
                pos = tf.reduce_sum(Q)
                neg = N-pos
                if  pos-N*ratio > -0.01:
                    print("warning: cannot achieve desired sample ratio, taking uniform")
                    P = Q*0+1
                else:
                    background_p = (1-ratio)*pos/(N*ratio-pos)
                    P = (Q + background_p)
                    P = P / tf.reduce_max(P)


                idx = np.argwhere( (P > np.random.uniform(low=0,high=1,size=(N))))
                R = tf.gather(R,idx[:,0],axis=0)
                R = R + np.random.uniform(low=0,high=1,size=R.shape)
                R = R/sz
                points.append(R)
                cnt = cnt + R.shape[0]
                if cnt >= M:
                    break

            if cnt < M:
                print("warning: cannot achieve desired sample ratio, taking uniform (2)")
               
                return None
                
            points = tf.concat(points,0)
            points = points[0:M,:]
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
          
      
      for k in range(nD):
          idx = local_boxes[:,k]<0
          local_boxes[idx,nD+k] = local_boxes[idx,nD+k] - local_boxes[idx,k]
          local_boxes[idx,k] = 0
          idx = local_boxes[:,k+nD]>1
          local_boxes[idx,k] = local_boxes[idx,k] - local_boxes[idx,k+nD] + 1
          local_boxes[idx,nD+k] = 1
      local_boxes = tf.convert_to_tensor(local_boxes,dtype=self.ftype)
            
      return local_boxes
  
    

    
  

  # Computes normalized coordinates of random crops
  #  bbox_sz - a list of size 4 (2D) or 6 (3D) representing the template of the box
  #              with origin zero.
  #  overlap - an integer giving the additoinal number of boxes per dimension
  #  jitter - to add random jitter 
  # return boxes of shape [numboxes,4] (2D) or [numboxes,6] (3D) 
  def tree_boxes(self,bbox_sz,overlap,jitter=0,jitter_border_fix=False):

      nD = self.ndim
      centers = [None] * nD
      nums = [None] * nD
      totnum = 1
      for k in range(nD):
        delta = bbox_sz[nD+k]-bbox_sz[k]
        nums[k] = np.floor(1/delta)+1+overlap
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
      return qwq, qwq.shape[0];


  def createCropsLocal(self,data_parent,labels_parent,crops,patch_size,this_scale_fac,generate_type,test,
      jitter=0,
      jitter_border_fix=False,
      num_patches=1,
      branch_factor=1,
      overlap=0,resolution=None,balance=None,verbose=True):
      
      scale_fac = self.scale_fac
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

        if crops is None and isinstance(init_scale,str):
            assert (resolution is not None), "for absolute init_scale you have to pass resolution"
            sizes_mm = init_scale.replace("mm","").split(",")
            sizes_mm = list(map(int, sizes_mm))
            bbox_sz = [0] * (nD*2)            
            for d in range(nD):
                sfac = sizes_mm[d]/resolution[d]/sz[d+1]
                bbox_sz[d]    = -sfac*0.5
                bbox_sz[d+nD] = sfac*0.5
                forwarded_aspects[d] = 1 #sfac*sz[d+1]
            forwarded_aspects = forwarded_aspects / np.amax(forwarded_aspects)
            
            
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
          local_boxes = self.random_boxes(bbox_sz,labels_parent,bsize*replicate_patches,balance)
        elif generate_type == 'tree':   
          if crops is not None:
              local_boxes = []
              for t in range(crops['parent_boxes'].shape[0]):
                lb,replicate_patches = self.tree_boxes(bbox_sz,overlap,jitter=jitter,jitter_border_fix=jitter_border_fix)
                local_boxes.append(tf.expand_dims(lb,1))
              local_boxes = tf.concat(local_boxes,1)
          else:
              local_boxes,replicate_patches = self.tree_boxes(bbox_sz,overlap,jitter=jitter,jitter_border_fix=jitter_border_fix)
            
          



    
      ############### compute box indices
      if crops is None: # the first layer
        parent_boxes = local_boxes
      else: # preceding layers (crops have to recomputed according to grandparent)
        last_boxes = crops['parent_boxes']
        if generate_type == 'random':     
          rans = [None] * (2*nD)          
          last_boxes = tf.tile(last_boxes,[replicate_patches,1])                   
          for k in range(nD):
              delta = last_boxes[:,(k+nD):(k+nD+1)]-last_boxes[:,k:(k+1)] 
              rans[k] = local_boxes[:,k:(k+1)]*delta + last_boxes[:,k:(k+1)]
              rans[k+nD] = local_boxes[:,(k+nD):(k+nD+1)]*delta + last_boxes[:,k:(k+1)]
          parent_boxes = tf.concat(rans,1)

        elif generate_type == 'tree':          
          last_boxes = tf.expand_dims(last_boxes,0)
          rans = [None] * (2*nD)
          for k in range(nD):
              delta = (last_boxes[:,:,(k+nD):(k+nD+1)]-last_boxes[:,:,k:(k+1)]) 
              rans[k] = local_boxes[:,:,k:(k+1)]*delta + last_boxes[:,:,k:(k+1)]
              rans[k+nD] = local_boxes[:,:,(k+nD):(k+nD+1)]*delta + last_boxes[:,:,k:(k+1)]
          parent_boxes = tf.concat(rans,2)
          parent_boxes = tf.reshape(parent_boxes,[parent_boxes.shape[0]*parent_boxes.shape[1], 2*nD])
          local_boxes = tf.reshape(local_boxes,[local_boxes.shape[0]*local_boxes.shape[1], 2*nD])
      

      dest_full_size = [None]*(nD+1)
      for k in range(nD):
          dest_full_size[k+1] = tf.convert_to_tensor(np.math.floor(patch_size[k]/np.min(parent_boxes[:,nD+k]-parent_boxes[:,k])),dtype=self.ftype)


      # compute the index suitable for gather_nd
      local_box_index,_ = self.convert_to_gatherND_index(local_boxes,sz,patch_size,interp_type=self.interp_type)
      parent_box_index,_ = self.convert_to_gatherND_index(parent_boxes,data_parent.shape,patch_size,interp_type=self.interp_type)
      parent_box_scatter_index, _ = self.convert_to_gatherND_index(parent_boxes,dest_full_size,patch_size,interp_type=self.scatter_type)
  


      resolution = []
      for k in range(nD):
        resolution.append(((parent_boxes[0,k+nD]-parent_boxes[0,k])*data_parent.shape[k+1]/patch_size[k]).numpy() ) 
                    
  
      if verbose:
        print("--------- cropping ")
        print("shape of patch: ", *patch_size )
        print("voxsize (relative to original scale): ", *resolution)
        print("numpatches in level: %d" % (parent_box_index.shape[0] / data_parent.shape[0]))
        print("shape of full output: ",  *list(map(lambda x: x.numpy(), dest_full_size[1:])))
        



      ############## do the actual cropping
      res_data = self.crop(data_parent,parent_box_index,resolution[0]*self.smoothfac_data,interp_type=self.interp_type)        
      if labels_parent is not None:      
        res_labels = self.crop(labels_parent,parent_box_index,resolution[0]*self.smoothfac_label,interp_type=self.interp_type)
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








