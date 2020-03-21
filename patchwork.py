# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# pip install tensorflow==2.1.0 matplotlib pillow opencv-python tensorflow_probability


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



########## Cropmodel ##########################################

def gaussian2D(std):  
  size = tf.cast(tf.math.floor(2.5*std),tf.int32).numpy()
  X,Y = tf.meshgrid(list(range(-size,size)),list(range(-size,size)))
  X = tf.cast(X,dtype=tf.float32)
  Y = tf.cast(Y,dtype=tf.float32)
  gauss_kernel = tf.exp(-(X*X + Y*Y)/(2*std*std))
  return gauss_kernel / tf.reduce_sum(gauss_kernel)

def conv_gauss2D(img,std):
  g = gaussian2D(std)
  g = tf.expand_dims(g,2)
  g = tf.expand_dims(g,3)
  r = []
  for k in range(img.shape[3]):
    r.append(tf.nn.conv2d(img[:,:,:,k:k+1],g,1,'SAME'))
  r = tf.concat(r,3)
  return r

def gaussian3D(std):  
  size = tf.cast(tf.math.floor(2.5*std),tf.int32).numpy()
  X,Y,Z = tf.meshgrid(list(range(-size,size)),list(range(-size,size)),list(range(-size,size)))
  X = tf.cast(X,dtype=tf.float32)
  Y = tf.cast(Y,dtype=tf.float32)
  Z = tf.cast(Y,dtype=tf.float32)
  gauss_kernel = tf.exp(-(X*X + Y*Y + Z*Z)/(2*std*std))
  return gauss_kernel / tf.reduce_sum(gauss_kernel)

def conv_gauss3D(img,std):
  g = gaussian3D(std)
  g = tf.expand_dims(g,3)
  g = tf.expand_dims(g,4)
  r = []
  for k in range(img.shape[4]):
    r.append(tf.nn.conv3d(img[:,:,:,:,k:k+1],g,[1,1,1,1,1],'SAME'))
  r = tf.concat(r,4)
  return r



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
    if self.intermediate_loss:
      for x in self.scales:
        out.append(x['labels_cropped'])
    else:
        out.append(self.scales[-1]['labels_cropped'])
    return out


  # stitched results (output of network) back into full image
  def stitchResult(self,r,level):
    qq = r[level]
    numlabels = qq.shape[-1]
    sc = self.scales[level]
    pbox_index = sc['parent_box_scatter_index']
    sha = list(sc['dest_full_size'])
    sha.append(numlabels)
    sha = sha[1:]
    return tf.scatter_nd(pbox_index,qq,sha), tf.scatter_nd(pbox_index,qq*0+1,sha);



class CropGenerator():

  def __init__(self,patch_size = (64,64),  # this is the size of our patches
                    scale_fac = 0.4,       # scale factor from one scale to another
                    init_scale = -1,       # if init_scale = -1, already first layer consists of patches, 
                                           # if init_scale = sfac, it's the scale factor which the full image is scaled to
                                           # if init_scale = [sx,sy], it's the shape the full image is scaled to
                    overlap = 4,           # overlap of patches in percent
                    depth=3,               # depth of patchwork
                    ndim=2,
                    ):
    self.model = None
    self.patch_size = patch_size
    self.scale_fac = scale_fac
    self.init_scale = init_scale
    self.depth = depth
    self.ndim = ndim
    self.overlap_perc = overlap # for serizalization
    self.overlap = 1+tf.constant(overlap,dtype=tf.float32)/100 + 0.000001

    assert scale_fac < 1 and scale_fac > 0.01, "please choose scale_fac in the interval (0.01, 1)"
    assert overlap >= 0 and overlap < 50 , "please choose overlap in the interval (0,50) "



  # generates cropped data structure
  # input:
  #   trainset.shape = [batch_dim,w,h,(d),f0]
  #   labelset.shape = [batch_dim,w,h,(d),f1]
  #  trainset and label set may also be list of tensors, which is needed
  #  when  dimensions of examples differ. But note that dimensions are 
  #  only allowed to differ, if init_scale=-1 .
  # output:
  #   a list of levels (see createCropsLocal for content)
  def sample(self,trainset,labelset,test=False,
             generate_type='random',  # 'random' or 'tree' or 'tree_full'
             num_patches=1,           #  if 'random' this gives the number of draws, otherwise no function
             randfun=None,            #  if 'tree' this is the funtion which computes the random jitter (e.g. lambda s : tf.random.normal(s,stddev=0.05)
             verbose=False):


    reptree = False
    if generate_type == 'tree_full':
      generate_type = 'tree'
      reptree = True



    if not isinstance(trainset,list):
      trainset = [trainset]
      labelset = [labelset]

    N = len(trainset)

    pool = []

    for j in range(N):

      x = self.createCropsLocal(trainset[j],labelset[j],None,generate_type,test,num_patches=num_patches,randfun=randfun,verbose=verbose)
      scales = [x]
      for k in range(self.depth-1):
        x = self.createCropsLocal(trainset[j],labelset[j],x,generate_type,test,num_patches=num_patches,randfun=randfun,verbose=verbose)
        scales.append(x)

      if reptree:
        scales = self.tree_complete(scales)
        

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

  # a simple resize of the image
  # 2D image array of size [w,h,f] (if batch_dim=False) or [b,w,h,f] (if batch_dim=True)
  # 3D image array of size [w,h,d,f] (if batch_dim=False) or [b,w,h,d,f] (if batch_dim=True)
  # dest_shape a list of new size (len(dest_shape) = 2 or 3)
  def resize(self,image,dest_shape,batch_dim=False):
      if not batch_dim:
        image = tf.expand_dims(image,0)
      nD = self.ndim      
      sz = image.shape
      rans = [None] * nD
      for k in range(nD):
        scfac = sz[k+1] / dest_shape[k]
        rans[k] = tf.cast(tf.range(dest_shape[k]),dtype=tf.float32)*scfac
      qwq = self.rep_rans(rans,dest_shape,nD)

      index = tf.dtypes.cast(qwq+0.5,dtype=tf.int32)
      res = []
      for i in range(sz[0]):
        res.append(tf.expand_dims(tf.gather_nd(tf.squeeze(image[i,...]),index),0))
      res = tf.concat(res,0)
      if len(res.shape) == 3:
        res = tf.expand_dims(res,3)
      if not batch_dim:
        res = tf.squeeze(res[0,...])

      return res
 
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
          
      if smoothfac is not None:
        data_smoothed =  conv_gauss(data_parent,tf.constant(smoothfac,dtype=tf.float32))
      else:
        data_smoothed =  data_parent
      ds0 = data_smoothed.shape[0]
      for k in range(repfac):
        res_data.append(tf.gather_nd(data_smoothed, parent_box_index[ds0*k:ds0*(k+1),...],batch_dims=1) )
      res_data = tf.concat(res_data,0)
      return res_data

  # to get a nice shape which dividable by divisor 
  def getClosestDivisable(self,x,divisor):
    for i in range(x,2*x):
      if i % divisor == 0:
        break
    return i

  # computes a meshgrid like thing, but  suitable for gather_nd
  # output shape 2D : w h 2  and 3D: w h d 3
  def rep_rans(self,rans,sizes,nD):
      if nD == 2:
        qwq = tf.concat( [       
                          tf.expand_dims(tf.tile(tf.expand_dims(rans[0],1),[1, sizes[1]]),2) ,
                          tf.expand_dims(tf.tile(tf.expand_dims(rans[1],0),[sizes[0], 1]),2) 
                          ] , 2) 
      elif nD == 3:        
        qwq = tf.concat( [       
                          tf.expand_dims(tf.tile(tf.expand_dims(tf.expand_dims(rans[0],1),2),[1, sizes[1], sizes[2]]),3) ,
                          tf.expand_dims(tf.tile(tf.expand_dims(tf.expand_dims(rans[1],0),2),[sizes[0], 1, sizes[2]]),3) ,
                          tf.expand_dims(tf.tile(tf.expand_dims(tf.expand_dims(rans[2],0),1),[sizes[0], sizes[1], 1]),3) 
                        ] , 3) 
      else: 
        assert "other than 2D/3D not implemented"
      return qwq

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
           sz[k+1] = tf.math.floor(patch_size[k]/(local_boxes[0,k+nD]-local_boxes[0,k]) )

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

      qwq = tf.expand_dims(self.rep_rans(rans,patch_size,nD),0)

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
  #  overlap - a float >=1 (e.g. 1.1 means 10% oversize)
  # returns boxes of shape [numboxes,4] (2D) or [numboxes,6] (3D) suitable as
  #  input for convert_to_gatherND_index
  def random_boxes(self,bbox_sz,numboxes,overlap):
      nD = self.ndim
      centers = [None]*(nD*2)
      for k in range(nD):
        c = tf.random.uniform(shape=(numboxes, 1))*(1-bbox_sz[k+nD]+bbox_sz[k])-bbox_sz[k]
        centers[k] = c
        centers[k+nD] = c
      local_boxes = tf.concat(centers,1) + bbox_sz*overlap
      return local_boxes

  # Computes normalized coordinates of random crops
  #  bbox_sz - a list of size 4 (2D) or 6 (3D) representing the template of the box
  #              with origin zero.
  #  overlap - a float >=1 (e.g. 1.1 means 10% oversize)
  #  randfun - to add random jitter 
  # return boxes of shape [numboxes,4] (2D) or [numboxes,6] (3D) 
  def tree_boxes(self,bbox_sz,overlap,randfun=None):

      nD = self.ndim
      centers = [None] * nD
      nums = [None] * nD
      totnum = 1
      for k in range(nD):
        delta = bbox_sz[nD+k]-bbox_sz[k]
        nums[k] = tf.floor(1/delta)+1
        frac = nums[k]*delta-1
        centers[k] = tf.cast(tf.range(nums[k]),dtype=tf.float32)*delta + (delta-frac)*0.5
        totnum *= nums[k]
      centers = self.rep_rans(centers,nums,nD)

      if randfun is not None:
        centers = centers + randfun(centers.shape)
      qwq = tf.tile(tf.reshape(centers,[totnum,nD]),[1,2]) + bbox_sz*overlap

      return qwq, qwq.shape[0];


  def createCropsLocal(self,data_parent,labels_parent,crops,generate_type,test,randfun=None,num_patches=1,verbose=True):
      patch_size = self.patch_size
      scale_fac = self.scale_fac
      init_scale = self.init_scale
      divisor = 8 # used for initital scale to get a nice image size
      overlap = self.overlap
      nD = self.ndim


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
      elif crops is None and init_scale != -1:               # the first layer is a isotropically scaled version
        patch_size = [0] * nD        
        qbox = [None] * (2*nD)
        for d in range(nD):
          desired = round(init_scale*sz[d+1])
          patch_size[d] = self.getClosestDivisable(desired,divisor)
          fac = patch_size[d]/desired
          qbox[d] = tf.zeros(shape=(replicate_patches*bsize, 1))
          qbox[d+nD] = fac*tf.ones(shape=(replicate_patches*bsize, 1))
        local_boxes = tf.concat(qbox,1)
      else:                                                  # the first layer is already patched
        asp = 0.5 * max(sz[1:nD])/max(patch_size)*scale_fac
        bbox_sz = [0] * (nD*2)
        for d in range(nD):
            bbox_sz[d]   = -asp*patch_size[d]/sz[d+1]
            bbox_sz[d+nD] = asp*patch_size[d]/sz[d+1]
        if generate_type == 'random':     
          local_boxes = self.random_boxes(bbox_sz,replicate_patches*bsize,overlap)
        elif generate_type == 'tree':          
          local_boxes,replicate_patches = self.tree_boxes(bbox_sz,overlap,randfun=randfun)



    
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
              delta = last_boxes[:,:,(k+nD):(k+nD+1)]-last_boxes[:,:,k:(k+1)] 
              rans[k] = local_boxes[:,:,k:(k+1)]*delta + last_boxes[:,:,k:(k+1)]
              rans[k+nD] = local_boxes[:,:,(k+nD):(k+nD+1)]*delta + last_boxes[:,:,k:(k+1)]
          parent_boxes = tf.concat(rans,2)
          parent_boxes = tf.reshape(parent_boxes,[parent_boxes.shape[0]*parent_boxes.shape[1], 2*nD])
          local_boxes =  local_boxes + 0*last_boxes
          local_boxes = tf.reshape(local_boxes,[local_boxes.shape[0]*local_boxes.shape[1], 2*nD])
      

      # compute the index suitable for gather_nd
      local_box_index,_ = self.convert_to_gatherND_index(local_boxes,sz,patch_size)
      parent_box_index,_ = self.convert_to_gatherND_index(parent_boxes,data_parent.shape,patch_size)
      parent_box_scatter_index, dest_full_size = self.convert_to_gatherND_index(parent_boxes,None,patch_size)


      resolution = []
      for k in range(nD):
        resolution.append(((parent_boxes[0,k+nD]-parent_boxes[0,k])*data_parent.shape[k+1]/patch_size[k]).numpy() ) 
                    
  
      if verbose:
        print("------------------------")
        print("shape of patch: ", *patch_size )
        print("voxsize (relative to original scale): ", *resolution)
        print("numpatches in level: %d" % (parent_box_index.shape[0] / data_parent.shape[0]))
        print("shape of full output: ",  *list(map(lambda x: x.numpy(), dest_full_size[1:])))



      ############## do the actual cropping
      res_data = self.crop(data_parent,parent_box_index,resolution[0]/2.0)        
      if labels_parent is not None:      
        res_labels = self.crop(labels_parent,parent_box_index,resolution[0])
      else:
        res_labels = None

      # for testing
      if test:
        if crops is None:
          images = tf.tile(images,[num_patches,1,1,1])
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
              }


  def scatter_valid(self, index, data, size):
      return tf.scatter_nd(index,data,size) 


  def testtree(self,im):
    randfun = lambda shape : tf.random.normal(shape,stddev=0.05)
    print(im.shape)
    c = self.sample(im[0:1,...],None,test=False,generate_type='tree',randfun=randfun,verbose=True)
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
  
  def apply_on_nifti(self,fname, ofname,
                 generate_type='tree',
                 jitter=0.05,
                 repetitions=5):
      nD = self.cropper.ndim
      
      img1 = nib.load(fname)        
      a = np.expand_dims(np.squeeze(img1.get_fdata()),0)
      if len(a.shape) < nD+2:
          a = np.expand_dims(a,nD+1)
      a = tf.convert_to_tensor(a,dtype=tf.float32)
      res = self.apply_full(a,generate_type=generate_type,
                            jitter=jitter,
                            repetitions=repetitions,
                            scale_to_original=True)

      pred_nii = nib.Nifti1Image(res, img1.affine, img1.header)
      nib.save(pred_nii,ofname)


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
    model = PatchWorkModel(cropper, blkCreator,**x)
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
        





