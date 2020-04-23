#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 15:37:03 2020

@author: reisertm
"""

import tensorflow as tf
import matplotlib.pyplot as plt
from . improc_utils import *


########## Cropmodel ##########################################


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
    

    # if self.intermediate_loss:
    #   for x in self.scales:
    #     if self.cropper.model.classifier_train:
    #         out.append(x['class_labels'])
    #     if self.cropper.model.spatial_train:          
    #         out.append(x['labels_cropped'])
    # else:
    #     if self.cropper.model.classifier_train:
    #         out.append(self.scales[-1]['class_labels'])
    #     if self.cropper.model.spatial_train:
    #         out.append(self.scales[-1]['labels_cropped'])
    
    
    
    return out


  # stitched results (output of network) back into full image
  def stitchResult(self,r,level, subselections = None):
    qq = r[level]
    numlabels = qq.shape[-1]
    sc = self.scales[level]
    pbox_index = sc['parent_box_scatter_index']
    if subselections is not None:
        s = subselections[level]
        pbox_index = tf.gather(pbox_index,s,axis=0)
    sha = list(sc['dest_full_size'])
    sha.append(numlabels)
    sha = sha[1:]
    return tf.scatter_nd(pbox_index,qq,sha), tf.scatter_nd(pbox_index,qq*0+1,sha);



class CropGenerator():

  def __init__(self,patch_size = (64,64),  # this is the size of our patches
                    scale_fac = 0.4,       # scale factor from one scale to another
                    init_scale = -1,       # a) if init_scale = -1, already first layer consists of patches, 
                                           # b) if init_scale = sfac, it's the scale factor which the full image is scaled to
                                           # c) if init_scale = [sx,sy], it's the shape the full image is scaled to
                    keepAspect = True,     # in case of c) this keeps the apsect ratio (also if for b) if shape is not a nice number)                                     
                    smoothfac_data = 0.5,  # 
                    smoothfac_label = 1.0, #
                    create_indicator_classlabels= False,
                    depth=3,               # depth of patchwork
                    ndim=2,
                    ):
    self.model = None
    self.patch_size = patch_size
    self.scale_fac = scale_fac
    self.smoothfac_data = smoothfac_data
    self.smoothfac_label = smoothfac_label
    self.init_scale = init_scale
    self.keepAspect = keepAspect
    self.create_indicator_classlabels = create_indicator_classlabels
    self.depth = depth
    self.ndim = ndim

    assert scale_fac < 1 and scale_fac > 0.01, "please choose scale_fac in the interval (0.01, 1)"

  def serialize_(self):
      return { 'patch_size':self.patch_size,
               'scale_fac' :self.scale_fac,
               'init_scale':self.init_scale,
               'smoothfac_data':self.smoothfac_data,
               'smoothfac_label':self.smoothfac_label,
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
             num_patches=1,           #  if 'random' this gives the number of draws, otherwise no function
             jitter=0,                #  if 'tree' this is the amount of random jitter
             overlap=0,
             augment=None,
             test=False,
             verbose=False):


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

      trainset_ = trainset[j]
      resolution_ = None
      if resolutions is not None:
          resolution_ = resolutions[j]
      
      if augment is not None:
          print("augmenting ...")
          trainset_,labels_ = augment(trainset_,labels_)
          print("augmenting done. ")

      if isinstance(self.patch_size,list):
          psize = self.patch_size[0]
      else:
          psize = self.patch_size

      x = self.createCropsLocal(trainset_,labels_,None,psize,generate_type,test,num_patches=num_patches,jitter=jitter,overlap=overlap,resolution=resolution_,verbose=verbose)
      if self.create_indicator_classlabels and x['labels_cropped'] is not None:
          x['class_labels'] = tf.expand_dims(tf.math.reduce_max(x['labels_cropped'],list(range(1,self.ndim+2))),1)
      else:
          x['class_labels'] = class_labels_
      
      scales = [x]
      for k in range(self.depth-1):
        if isinstance(self.patch_size,list):
          psize = self.patch_size[k+1]
        else:
          psize = self.patch_size
        x = self.createCropsLocal(trainset_,labels_,x,psize,generate_type,test,num_patches=num_patches,jitter=jitter,overlap=overlap,resolution=resolution_,verbose=verbose)
        if self.create_indicator_classlabels and x['labels_cropped'] is not None:
            x['class_labels'] = tf.expand_dims(tf.math.reduce_max(x['labels_cropped'],list(range(1,self.ndim+2))),1)
        else:
            x['class_labels'] = class_labels_
        scales.append(x)

      if reptree:
        scales = self.tree_complete(scales)

      if self.model.classifier_train and scales[k]['class_labels'] is not None:
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
  def crop(self,data_parent,parent_box_index,smoothfac=None):
      repfac = parent_box_index.shape[0] // data_parent.shape[0]
      res_data = []
      if self.ndim == 2:
          conv_gauss = conv_gauss2D
      elif self.ndim == 3:
          conv_gauss = conv_gauss3D
          
      if smoothfac is not None and smoothfac > 0.0:
        data_smoothed =  conv_gauss(data_parent,tf.constant(smoothfac,dtype=tf.float32))
      else:
        data_smoothed =  data_parent
      ds0 = data_smoothed.shape[0]
      for k in range(repfac):
        res_data.append(tf.gather_nd(data_smoothed, parent_box_index[ds0*k:ds0*(k+1),...],batch_dims=1) )
      res_data = tf.concat(res_data,0)
      return res_data


  # Converts normalized coordinates (as used in tf.crop_and_resize) in local_boxes 
  # to actual pixel coordinates, it assumed that all boxes are of the same size!! 
  #  local_boxes - normalized coordinates of shape [N,4] (2D) or [N,6] (3D), 
  #  sz - abs. size of images from which crop is performed (list of size nD)
  #  patch_size - size of patches (list of size nD) 
  def convert_to_gatherND_index(self,local_boxes,sz,patch_size):
      nD = self.ndim

      if sz is None:             
         sz = [None] * (nD+1)
         for k in range(nD):
           sz[k+1] = tf.math.round(patch_size[k]/(local_boxes[0,k+nD]-local_boxes[0,k]) )

      rans = [None] * nD
      start_abs = [None] * nD
      for k in range(nD):
        scfac = (local_boxes[0,(k+nD):(k+nD+1)]-local_boxes[0,k:k+1])/patch_size[k]*sz[k+1]
        rans[k] = tf.cast(tf.range(patch_size[k]),dtype=tf.float32)*scfac
        start_abs[k] = local_boxes[:,k:(k+1)] * sz[k+1]

      start_abs = tf.transpose(tf.math.floor(start_abs),[1,0,2]);
      res_shape = [1] * (nD+2)
      res_shape[0] = local_boxes.shape[0]
      res_shape[nD+1] = nD
      start_abs = tf.reshape(start_abs,res_shape)

      qwq = tf.expand_dims(rep_rans(rans,patch_size,nD),0)

      local_box_index = tf.dtypes.cast(start_abs+qwq+0.5,dtype=tf.int32)


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
  def random_boxes(self,bbox_sz,numboxes):
      nD = self.ndim
      centers = [None]*(nD*2)
      # for k in range(nD):
      #   c = tf.random.uniform(shape=(numboxes, 1))*(1-bbox_sz[k+nD]+bbox_sz[k])-bbox_sz[k]
      #   centers[k] = c
      #   centers[k+nD] = c
      # local_boxes = tf.concat(centers,1) + bbox_sz      
      # return local_boxes
      
      for k in range(nD):
        c = np.random.uniform(0,1,(numboxes, 1))
        centers[k] = c
        centers[k+nD] = c
      local_boxes = np.concatenate(centers,1) + bbox_sz
      
      for k in range(nD):
          idx = local_boxes[:,k]<0
          local_boxes[idx,nD+k] = local_boxes[idx,nD+k] - local_boxes[idx,k]
          local_boxes[idx,k] = 0
          idx = local_boxes[:,k+nD]>1
          local_boxes[idx,k] = local_boxes[idx,k] - local_boxes[idx,k+nD] + 1
          local_boxes[idx,nD+k] = 1
      local_boxes = tf.convert_to_tensor(local_boxes,dtype=tf.float32)
            
      return local_boxes

  # Computes normalized coordinates of random crops
  #  bbox_sz - a list of size 4 (2D) or 6 (3D) representing the template of the box
  #              with origin zero.
  #  overlap - an integer giving the additoinal number of boxes per dimension
  #  jitter - to add random jitter 
  # return boxes of shape [numboxes,4] (2D) or [numboxes,6] (3D) 
  def tree_boxes(self,bbox_sz,overlap,jitter=0):

      nD = self.ndim
      centers = [None] * nD
      nums = [None] * nD
      totnum = 1
      for k in range(nD):
        delta = bbox_sz[nD+k]-bbox_sz[k]
        nums[k] = np.floor(1/delta)+1+overlap
        delta_small= (1-delta)/(nums[k]-1) -0.000001
        frac = nums[k]*delta-1
        centers[k] = tf.cast(tf.range(nums[k]),dtype=tf.float32)*delta_small + delta*0.5
        totnum *= nums[k]
      centers = rep_rans(centers,nums,nD)
      
      if jitter > 0:
          sh = centers.shape
          rands = []
          for k in range(nD):
             delta = bbox_sz[nD+k]-bbox_sz[k]              
             rng = np.random.uniform(-delta/2*jitter,delta/2*jitter,sh[0:nD])
             if nD ==2:
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
             rands.append(np.expand_dims(rng,nD))
          rands = np.concatenate(rands,nD)
          centers = centers + rands

      qwq = tf.tile(tf.reshape(centers,tf.cast([totnum,nD],dtype=tf.int32)),[1,2]) + bbox_sz
      return qwq, qwq.shape[0];


  def createCropsLocal(self,data_parent,labels_parent,crops,patch_size,generate_type,test,jitter=0,num_patches=1,overlap=0,resolution=None,verbose=True):
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
        if generate_type == 'random':     # this is used for training
           replicate_patches = num_patches
        elif generate_type == 'tree':     # this we need for apply
          replicate_patches = 1
        else:
          assert 'not valid generate_type'
      else:
        images = crops['data_cropped']
        if generate_type == 'random':     # this is used for training
          replicate_patches = 1
        elif generate_type == 'tree':     # this we need for apply
          replicate_patches = None
        else:
          assert 'not valid generate_type'
      sz = images.shape
      bsize = sz[0] 
      
      if generate_type == 'tree' and crops is None:
        assert bsize == 1, "for generate_type=tree the batch_size has to be one!"
 

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
                asp.append(sz[d+1]/patch_size[d]*scale_fac*0.5)
            asp = max(asp)
            bbox_sz = [0] * (nD*2)
            for d in range(nD):
                bbox_sz[d]   = -asp*patch_size[d]/sz[d+1]* aspect_correction[d]
                bbox_sz[d+nD] = asp*patch_size[d]/sz[d+1] *aspect_correction[d]


        if generate_type == 'random':     
          local_boxes = self.random_boxes(bbox_sz,replicate_patches*bsize)
        elif generate_type == 'tree':          
          local_boxes,replicate_patches = self.tree_boxes(bbox_sz,overlap,jitter=jitter)



    
      ############### compute box indices
      if crops is None: # the first layer
        parent_boxes = local_boxes
      else: # preceding layers (crops have to recomputed according to grandparent)
        last_boxes = crops['parent_boxes']
        if generate_type == 'random':     
          rans = [None] * (2*nD)
          for k in range(nD):
              delta = last_boxes[:,(k+nD):(k+nD+1)]-last_boxes[:,k:(k+1)] 
              rans[k] = local_boxes[:,k:(k+1)]*delta + last_boxes[:,k:(k+1)]
              rans[k+nD] = local_boxes[:,(k+nD):(k+nD+1)]*delta + last_boxes[:,k:(k+1)]
          parent_boxes = tf.concat(rans,1)

        elif generate_type == 'tree':          
          last_boxes = tf.expand_dims(last_boxes,0)
          local_boxes = tf.expand_dims(local_boxes,1)
          rans = [None] * (2*nD)
          for k in range(nD):
              delta = (last_boxes[:,:,(k+nD):(k+nD+1)]-last_boxes[:,:,k:(k+1)]) 
              rans[k] = local_boxes[:,:,k:(k+1)]*delta + last_boxes[:,:,k:(k+1)]
              rans[k+nD] = local_boxes[:,:,(k+nD):(k+nD+1)]*delta + last_boxes[:,:,k:(k+1)]
          parent_boxes = tf.concat(rans,2)
          parent_boxes = tf.reshape(parent_boxes,[parent_boxes.shape[0]*parent_boxes.shape[1], 2*nD])
          local_boxes =  tf.tile(local_boxes,[1,last_boxes.shape[1],1])
          local_boxes = tf.reshape(local_boxes,[local_boxes.shape[0]*local_boxes.shape[1], 2*nD])
      

      # compute the index suitable for gather_nd
      local_box_index,_ = self.convert_to_gatherND_index(local_boxes,sz,patch_size)
      parent_box_index,_ = self.convert_to_gatherND_index(parent_boxes,data_parent.shape,patch_size)
      parent_box_scatter_index, dest_full_size = self.convert_to_gatherND_index(parent_boxes,None,patch_size)
  
      for k in range(nD):
          dest_full_size[k+1] = tf.convert_to_tensor(np.math.floor(patch_size[k]/np.min(parent_boxes[:,nD+k]-parent_boxes[:,k])),dtype=tf.int32)


      resolution = []
      for k in range(nD):
        resolution.append(((parent_boxes[0,k+nD]-parent_boxes[0,k])*data_parent.shape[k+1]/patch_size[k]).numpy() ) 
                    
  
      if verbose:
        print("------------------------")
        print("shape of patch: ", *patch_size )
        print("voxsize (relative to original scale): ", *resolution)
        print("numpatches in level: %d" % (parent_box_index.shape[0] / data_parent.shape[0]))
        print("shape of full output: ",  *list(map(lambda x: x.numpy(), dest_full_size[1:])))
   #     print("shape of full output: ",   dest_full_size[1:])



      ############## do the actual cropping
      res_data = self.crop(data_parent,parent_box_index,resolution[0]*self.smoothfac_data)        
      if labels_parent is not None:      
        res_labels = self.crop(labels_parent,parent_box_index,resolution[0]*self.smoothfac_label)
      else:
        res_labels = None

      # for testing
      if test:
        if crops is None:
          mult = (nD+2)*[1]
          mult[0] = num_patches
          images = tf.tile(images,mult)
       # test = tf.image.crop_and_resize(images,local_boxes,box_indices,patch_size)
        test = tf.gather_nd(images,local_box_index,batch_dims=1)
      else:
        test = None


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



  def scatter_valid(self, index, data, size):
      return tf.scatter_nd(index,data,size) 


  def testtree(self,im):
    print(im.shape)
    c = self.sample(im[0:1,...],None,test=False,generate_type='tree',jitter=0.1,verbose=True)
    self.showtest(c)

  def testrandom(self,im):
    c = self.sample(im[0:1,...],None,test=False,generate_type='random',verbose=True,num_patches=300)
    self.showtest(c)
 
  def showtest(self,c):
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








