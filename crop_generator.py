#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 15:37:03 2020

@author: reisertm
"""

import tensorflow as tf
from timeit import default_timer as timer
from collections.abc import Iterable
import math

from .improc_utils import *



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
    
    self.scales[0]['age'] = tf.zeros([self.num_patches()])
    self.attrs = ['data_cropped','local_box_index','labels_cropped','class_labels','age']

  def extb2dim(self,x,batchdim2):
      if batchdim2 == -1:
          return x      
      sz = x.shape
      x = tf.reshape(x,[sz[0]//batchdim2, batchdim2] + sz[1:])
      return x

  def num_patches(self):
      return self.scales[0]['data_cropped'].shape[0]
  
  def merge(self,ci):
      for i in range(len(self.scales)):
          x = self.scales[i]
          y = ci.scales[i]
          for k in self.attrs:
              if k in x and x[k] is not None and k != 'dest_full_size':
                  x[k] = tf.concat([x[k],y[k]],0)
      self.scales[0]['age'] += 1

  def getAge(self):
      return self.scales[0]['age']

  def subset(self,patchloss,fraction,maxage=1000):
      patchloss = tf.where(patchloss[:,1:2]>maxage,0,patchloss[:,0:1])
      n = tf.cast(patchloss.shape[0]*fraction,dtype=tf.int32)
      idx = tf.argsort(patchloss,0,'DESCENDING')[0:n]
      for i in range(len(self.scales)):
          x = self.scales[i]
          for k in self.attrs:
              #print(k)
              if k in x and x[k] is not None and k != 'dest_full_size':
                 #print(x[k].shape)
                 x[k] = tf.gather(x[k],idx[:,0]) 
      


  # get input training data 
  def getInputData(self,sampletyp=None):                    
    if sampletyp == None:
        sampletyp = [None,-1]
    batchdim2 = sampletyp[1]
    cnt = 0
    inp = {}
    for x in self.scales:
      inp['input'+str(cnt)] = self.extb2dim(x['data_cropped'],batchdim2)
      inp['cropcoords'+str(cnt)] = self.extb2dim(x['local_box_index'],batchdim2)
      cnt = cnt + 1
    inp['batchindex'] = tf.range(inp['input0'].shape[0])

    if sampletyp[0] is not None:      
        cnt = 0
        for x in self.scales:
          inp['input'+str(cnt)] = tf.gather(inp['input'+str(cnt)] ,sampletyp[0],axis=0)
          inp['cropcoords'+str(cnt)] = tf.gather(inp['cropcoords'+str(cnt)],sampletyp[0],axis=0)
          cnt = cnt + 1

    return inp

  def getInputDataset(self,sampletyp=None):                    
      a = self.getInputData(sampletyp)
      return tf.data.Dataset.from_tensor_slices(a)

  def getDataset(self,sampletyp=None):                    
      a = self.getInputData(sampletyp)
      b = self.getTargetData(sampletyp)
      targetset= tf.data.Dataset.zip(tuple(map(tf.data.Dataset.from_tensor_slices,b)))
      return tf.data.Dataset.zip((tf.data.Dataset.from_tensor_slices(a),targetset))


  # get target training data 
  def getTargetData(self,sampletyp=None):
    if sampletyp == None:
        sampletyp = [None,-1]
    batchdim2 = sampletyp[1]
        
    out = []    
    create_indicator_classlabels = False
    classifier_train = False
    spatial_train = True
    intermediate_loss = True
    cls_intermediate_loss= False
    spatial_max_train = False
    
    if self.cropper.model is not None:    
        create_indicator_classlabels = self.cropper.create_indicator_classlabels
        classifier_train = self.cropper.model.classifier_train
        spatial_train = self.cropper.model.spatial_train
        intermediate_loss = self.cropper.model.intermediate_loss
        cls_intermediate_loss= self.cropper.model.cls_intermediate_loss
        spatial_max_train = self.cropper.model.spatial_max_train
    
    
    depth = len(self.scales)
    for i in range(depth-1):
        x = self.scales[i]
        if cls_intermediate_loss and classifier_train:
            out.append(x['class_labels'])            
        if intermediate_loss and spatial_train:
            out.append(x['labels_cropped'])
    if spatial_train and not spatial_max_train:
        out.append(self.scales[-1]['labels_cropped'])


    if batchdim2 != -1:
        for k in range(len(out)):
            out[k] = self.extb2dim(out[k],batchdim2)
    #        out[k] = tf.reduce_max(out[k],axis=1)

    if classifier_train or spatial_max_train:
        out = [self.scales[-1]['class_labels']] + out
    

    if sampletyp[0] is not None:      
        for k in range(len(out)):
            out[k] = tf.gather(out[k],sampletyp[0],axis=0)
        
    return out

  def stitchResult(self,r,level):
     return stitchResult(r,level,self.scales,self.cropper.scatter_type)

# stitched results (output of network) back into full image
def stitchResult(r,level, scales,scatter_type):
    qq = r[level]
    numlabels = qq.shape[-1]
    sc = scales[level]
    pbox_index = sc['parent_box_scatter_index']
    sha = tf.concat([sc['dest_full_size'],[numlabels]],0)
    sha = tf.cast(sha,dtype=pbox_index.dtype)
    if scatter_type=='NN':        
        return tf.scatter_nd(pbox_index,qq,sha), tf.scatter_nd(pbox_index,qq*0+1,sha);
    else:
        return scatter_interp(pbox_index,qq,sha)

def stitchResult_withstride(r,level, scales,scatter_type,stride):
    qq = r[level]
    numlabels = qq.shape[-1]
    sc = scales[level]
    pbox_index = sc['parent_box_scatter_index']
    sha = list(sc['dest_full_size'])
    sha.append(numlabels)
    
    sz = pbox_index.shape
    pbox_index = tf.reshape(pbox_index,tf.TensorShape([sz[0]//stride,stride]).concatenate(sz[1:]) )
    sz = qq.shape
    qq = tf.reshape(qq,tf.TensorShape([sz[0]//stride,stride]).concatenate(sz[1:]) )

    sha[0] = sz[0]//stride
    
    
    idx = tf.range(0,sha[0])
    for k in range(0,len(sha)):
        idx = tf.expand_dims(idx,1)
    idx = tf.tile(idx,tf.TensorShape([1]).concatenate(pbox_index.shape[1:-1]).concatenate(1))
    pbox_index = tf.concat([idx,pbox_index],len(sha))
    
    if scatter_type=='NN':        
        return tf.scatter_nd(pbox_index,qq,sha)/(0.000001+tf.scatter_nd(pbox_index,qq*0+1,sha))
    else:
        assert False,'not yet implemented'

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

  def __init__(self,scheme=None,                    
                    patch_size = (64,64),  # (deprecated) this is the size of our patches
                    scale_fac = 0.4,       # (deprecated) scale factor from one scale to another
                    scale_fac_ref = 'max', # (deprecated) for inhomgenus matrixsizes scale_fac has to computed to certain axis (possible values 'max','min' or int refering to dimension)
                    init_scale = -1,       # (deprecated) a) if init_scale = -1, already first layer consists of patches, 
                                           # (deprecated) b) if init_scale = sfac, it's the scale factor which the full image is scaled to
                                           # (deprecated) c) if init_scale = [sx,sy], it's the shape the full image is scaled to
                    auto_patch = None,     # (deprecated)                                   
                    keepAspect = True,     # (deprecated)  in case of c) this keeps the apsect ratio (also if for b) if shape is not a nice number)                                     
                    
                    transforms = None,
                    snapper = None,
                    smoothfac_data = 0,  # 
                    smoothfac_label = 0, #
                    categorial_label = None,
                    interp_type = 'NN',    # nearest Neighbor (NN) or linear (lin)
                    scatter_type = 'NN',
                    normalize_input = None,
                    create_indicator_classlabels= False,
                    depth=3,               # depth of patchwork
                    ndim=2,
                    ftype=tf.float32
                    ):
    self.model = None
    self.scheme = scheme
    self.transforms = transforms
    self.patch_size = patch_size
    self.scale_fac = scale_fac
    self.scale_fac_ref = scale_fac_ref
    self.smoothfac_data = smoothfac_data
    self.smoothfac_label = smoothfac_label
    self.categorial_label = categorial_label
    self.interp_type = interp_type
    self.scatter_type = scatter_type
    self.init_scale = init_scale
    self.normalize_input = normalize_input
    self.keepAspect = keepAspect
    self.create_indicator_classlabels = create_indicator_classlabels
    self.depth = depth
    self.ndim = ndim
    self.ftype=ftype
    self.snapper = snapper
    self.dest_full_size = [None]*depth


      
            

    if auto_patch is not None:    

        psize = 32
        factor_thres = 0.75
        dest_factor = 1
        initial_scale = None
        voxsize = None
        if "psize" in auto_patch:
            psize = auto_patch['psize']
        if "factor_thres" in auto_patch:
            factor_thres = auto_patch['factor_thres']
        if "dest_factor" in auto_patch:
            dest_factor =auto_patch['dest_factor']
        if "initial_scale" in auto_patch:
            initial_scale = auto_patch['initial_scale']
        if "voxsize" in auto_patch:
            voxsize = auto_patch['voxsize']
            
        shape = auto_patch['shape']
        scfacs = []
        patch_size = []
        
        
        cf = lambda p,k : math.pow(dest_factor*p/shape[k],1/depth)            
        if initial_scale is not None and depth > 1:
            cf = lambda p,k : math.pow(dest_factor*p/shape[k]/initial_scale,1/(depth-1))

        for k in range(0,ndim):
            p = psize
            f = cf(p,k)
            while f > factor_thres and p>4:
                p = p//2
                f = cf(p,k)                    
            scfacs.append(f)
            patch_size.append(p)

            
        self.scale_fac = {}
 
        if initial_scale is not None and depth > 1:
            self.scale_fac['level0'] = [initial_scale]*ndim
            for k in range(1,depth):
                self.scale_fac['level'+str(k)] = scfacs
        else:
            for k in range(0,depth):
                self.scale_fac['level'+str(k)] = scfacs
            
            
        self.patch_size = patch_size
        self.depth = depth
        self.init_scale = -1
        self.scale_fac_ref =  "perdim"

        if voxsize is not None:
            first = [""]*ndim
            for k in range(0,ndim):
                first[k] = self.scale_fac['level0'][k]*shape[k]*voxsize[k]
            self.init_scale = ",".join(map(lambda x: str(x)+"mm",first))



  def get_patchsize(self,level):   # patch_size could be eg [32,32], or a list [ [32,32], [32,32] ] corresponding to differne levels
          pats = self.patch_size
          if self.scheme is not None and 'patch_size' in self.scheme:
              pats = self.scheme['patch_size']
          if isinstance(pats[0],Iterable):
              return pats[level]
          else: 
              return pats

  def get_outpatchsize(self,level):  
          pats = self.patch_size
          if self.scheme is not None and 'patch_size' in self.scheme:
              pats = self.scheme['patch_size']
          if self.scheme is not None and 'out_patch_size' in self.scheme:
              pats = self.scheme['out_patch_size']
          if isinstance(pats[0],Iterable):
              return pats[level]
          else: 
              return pats
        
  def get_scalefac(self,level):  # either a float, or a list of floats where each entry corresponds to a different depth level,
                                    # or dict with entries { 'level0' : [0.5,0.5] , 'level1' : [0.4,0.3]} where scalefac is dependent on dimension and level
            if isinstance(self.scale_fac,float) or isinstance(self.scale_fac,int):
                return [self.scale_fac]*self.ndim
            elif isinstance(self.scale_fac,dict):
                if ('level' + str(level))  not in self.scale_fac:
                    return None
                tmp = self.scale_fac['level' + str(level)]
                if isinstance(tmp,float) or isinstance(tmp,int):
                    return [tmp]*self.ndim
                else:
                    return tmp
            elif isinstance(self.scale_fac,Iterable):
                return [self.scale_fac[level]]*self.ndim
        
  def get_smoothing(self,level,which):
            if which == 'data':
                sm = self.smoothfac_data
            else:
                sm = self.smoothfac_label
            if isinstance(sm,list):
                return sm[level]
            else:
                if sm == 'globalmax' and which == 'label':
                    if level == self.depth-1:
                        return None
                    else:
                        return sm                    
                else:
                   return sm
  
  @staticmethod
  def getTransform(transform,qdir):
      f = None
      if transform == 'tensor':
          f = lambda a : CropGenerator.transform_behaviour_tensor(a,qdir)
      return f
      
  @staticmethod
  def transform_behaviour_tensor(orients,qdir):  
    tensor = lambda a : tf.cast(a,dtype=tf.float32)
    s = 1/tf.math.sqrt(2.0)
    T = tensor([
        [[1,0,0],[0,0,0],[0,0,0]],
        [[0,0,0],[0,1,0],[0,0,0]],
        [[0,0,0],[0,0,0],[0,0,1]],
        [[0,s,0],[s,0,0],[0,0,0]],
        [[0,0,s],[0,0,0],[s,0,0]],
        [[0,0,0],[0,0,s],[0,s,0]]])    
    q = tf.einsum('bxy,yf->bfx',orients[:,0:3,0:3],qdir)
    qq = tf.einsum('bfx,bfy,ixy->bif',q,q,T)
    return qq


  def serialize_(self):
      return { 'scheme':self.scheme,
               'patch_size':self.patch_size,
               'scale_fac' :self.scale_fac,
               'scale_fac_ref' :self.scale_fac_ref,
               'interp_type' :self.interp_type,
               'scatter_type' :self.scatter_type,
               'init_scale':self.init_scale,
               'snapper':self.snapper,
               'categorial_label':self.categorial_label,
               'smoothfac_data':self.smoothfac_data,
               'smoothfac_label':self.smoothfac_label,
               'normalize_input':self.normalize_input,
               'create_indicator_classlabels':self.create_indicator_classlabels,
               'keepAspect':self.keepAspect,
               'transforms':self.transforms,
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
             snapper=None,
             patch_size_factor=1,
             dphi=0,
             overlap=0,
             balance=None,
             augment=None,
             test=False,
             training=False,
             lazyEval=None,
             verbose=False):
      
      
    def at(dic,key):
        if key in dic:
            return dic[key]
        else:
            return None


           
    def extend_classlabels(x,class_labels_):
      if self.create_indicator_classlabels and x['labels_cropped'] is not None:
          tmp = tf.math.reduce_mean(x['labels_cropped'],list(range(1,self.ndim+1)))
          tmp = tf.cast(tmp>0.5,dtype=self.ftype)
          return tmp
      else:
          return class_labels_


    tensor = lambda a : tf.cast(a,dtype=self.ftype)


    classifier_train = False
    spatial_max_train = False
    spatial_train = True
    if self.model is not None:
        classifier_train = self.model.classifier_train
        spatial_max_train = self.model.spatial_max_train
        spatial_train = self.model.spatial_train
        
        
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
    
    balances = [None]*self.depth
    for k in range(self.depth):
        balances[k] = []

    if snapper is None:
        if self.snapper is None:
            snapper = [0] + [1]*(self.depth-1)
            self.snapper = snapper
        else:
            snapper = self.snapper


    # this the main loop running over the image list
    for j in range(N):
        
      if not verbose:
          if j%round(N/20+1)==0:
              print(("{:.0f}% ").format(100*j/N),end="" )
          
                
      # grep and prep the labels
      labels_ = None
      class_labels_ = None
      if labelset is not None and labelset[j] is not None:
          if (classifier_train and not self.create_indicator_classlabels) and self.model.spatial_train:
              class_labels_ = labelset[j][0]
              labels_ = labelset[j][1]
          elif spatial_train and not spatial_max_train:
              labels_ = labelset[j]
          else:
              class_labels_ = labelset[j]

     #     if self.categorial_label is not None:
     #         if labels_.dtype != 'float32':
     #            labels_ = tf.cast(labels_,tf.float32)

      # get the data 
      trainset_ = trainset[j]
      
      if self.normalize_input == 'ct':
         trainset_ = tf.math.log(tf.math.maximum(trainset_+200,0.001))
         trainset_ = tf.math.maximum(trainset_-4,0)/10
      elif self.normalize_input == 'ct2':
         trainset_ = tf.math.log(tf.math.maximum(trainset_+1000,0.001))
      elif self.normalize_input == 'max':
         trainset_ = trainset_/tf.reduce_max(trainset_,keepdims=True,axis=range(1,self.ndim+1))   
      elif self.normalize_input == 'mean':
         trainset_ = trainset_/tf.reduce_mean(trainset_,keepdims=True,axis=range(1,self.ndim+1))   
      elif self.normalize_input is not None:
         assert False,'not valid normalize_input'
          
          
      # if resolutions are also passed
      resolution_ = None
      if resolutions is not None:
          if isinstance(resolutions[j],dict):
              resolution_ = resolutions[j]
              if 'bval' in resolutions[j]:
                  qdir = tf.math.sqrt(tensor(resolutions[j]['bval'])/1000)*tensor(resolutions[j]['bvec'])              
                  
                  
                  
          else:
              resolution_ = resolutions[j]
              resolution_ = resolution_[0:self.ndim]
      
      input_transform_behaviour = None
      label_transform_behaviour = None
      if self.transforms is not None:
          input_transform_behaviour=CropGenerator.getTransform(self.transforms[0],qdir)
          label_transform_behaviour=CropGenerator.getTransform(self.transforms[-1],qdir)
      
      
      dphi1=dphi
      dphi2=dphi
      flip=None
      dscale = 0
      independent_augmentation=False
      pixel_noise = 0
      
      

      tensor = lambda a : tf.cast(a,dtype=self.ftype)
      int32 = lambda a : tf.cast(a,dtype=tf.int32)

      
      if augment is not None:                  
          if isinstance(augment,dict):
              if 'dphi' in augment:
                  dphi1 = tensor(augment['dphi'])*0.5
                  dphi2 = tensor(augment['dphi'])*0.5
              if 'dphi1' in augment:
                  dphi1 = tensor(augment['dphi1'])
              if 'dphi2' in augment:
                  dphi2 = tensor(augment['dphi2'])
              if 'flip' in augment:
                  flip = tensor(augment['flip'])
              if 'dscale' in augment:
                  dscale = tensor(augment['dscale'])
              if 'pixel_noise' in augment:
                  pixel_noise = tensor(augment['pixel_noise'])
              if 'independent_augmentation' in augment:
                  independent_augmentation = augment['independent_augmentation']
          else:
              print("augmenting ...")
              trainset_,labels_ = augment(trainset_,labels_)
              print("augmenting done. ")





      def getPatchingParams(input_width,input_shape,input_edges,resolution,depth):

          showten = lambda a,n: "["+ (",".join(map(lambda x: ("{:."+str(n)+"f}").format(x), a.numpy()))) + "]"

          input_voxsize = tensor(input_width)/(tensor(input_shape)-1);
          if verbose:
              print("input:  shape:"+ showten(tensor(input_shape),0)+
                    "  width(mm):"+showten(tensor(input_width),1) +
                    '  voxsize:' +  showten(input_voxsize,2) )
              print("snapper " + " ".join(str(x) for x in snapper))
    
          nD = len(resolution)        
          patch_shapes = list(map(lambda k: tensor(self.get_patchsize(k)),range(depth)))
          out_patch_shapes =list(map(lambda k: tensor(self.get_outpatchsize(k)),range(depth)))
        
          if self.scheme is not None:
              destvox_mm = at(self.scheme,'destvox_mm')
              destvox_rel = at(self.scheme,'destvox_rel')
              fov_mm = at(self.scheme,'fov_mm')
              fov_rel = at(self.scheme,'fov_rel')
                                
              if fov_mm is not None:
                  patch_widths = [tensor(fov_mm)]
              else:
                  patch_widths = [tensor(fov_rel)*input_width]
            
              if destvox_mm is not None:
                  final_width = (out_patch_shapes[-1]-1)*tensor(destvox_mm)
              else:
                  final_width = (out_patch_shapes[-1]-1)*input_width/(tensor(input_shape)-1)*tensor(destvox_rel)
              
              if depth == 1:
                  patch_widths = [tensor(final_width)]
              else:
                  fac = tf.math.pow(final_width / patch_widths[0],1/(depth-1))
                  for k in range(1,depth):
                      patch_widths.append(fac*patch_widths[-1])
                  
          else:
            
              
              init_scale = self.init_scale
              depth = self.depth
              if isinstance(init_scale,str) or init_scale == -1:
                 if isinstance(init_scale,str):
                    if init_scale.find('mm') != -1 or init_scale.find('cm') != -1:
                        assert (resolution is not None), "for absolute init_scale you have to pass resolution"
                        if init_scale.find('cm') != -1:
                            sizes_mm = init_scale.replace("cm","").split(",")
                            sizes_mm = list(map(lambda x: float(x)*10, sizes_mm))
                        else:
                            sizes_mm = init_scale.replace("mm","").split(",")
                            sizes_mm = list(map(float, sizes_mm))                   
                        patch_widths = [tensor(sizes_mm)]
                    else:
                       assert False, "bug in init_scale"
                 else:
                    patch_widths = []
                
                 shapes = [tensor(input_shape)] + patch_shapes
                 for k in range(len(patch_widths),depth):
                    asp = []
                    for d in range(nD):
                        asp.append(shapes[k][d]/shapes[k+1][d]*self.get_scalefac(k)[d])
                    if self.scale_fac_ref == 'max':
                        asp = [max(asp)]*nD
                    elif self.scale_fac_ref == 'min':
                        asp = [min(asp)]*nD
                    elif self.scale_fac_ref == 'perdim':
                        asp = asp
                    else:
                        asp = [asp[self.scale_fac_ref]]*nD
                    if len(patch_widths) == 0:
                        patch_widths.append(tensor(asp)* shapes[k+1]/shapes[k]*input_width)
                    else:
                        patch_widths.append(tensor(asp)* shapes[k+1]/shapes[k]*patch_widths[-1])
              
                
          # derive shape of output image
          dest_edges = []
          dest_shapes = []
          for k in range(len(out_patch_shapes)):
            w = patch_widths[k]/(out_patch_shapes[k]-1)*1
            dshape = int32(input_width/w)
            vsz =  input_width/tensor(dshape-1)
            dedge = tf.matmul(input_edges[0,:,:],tf.linalg.diag(tf.concat([vsz/input_voxsize,[1]],0)))
            #dedge = tf.linalg.diag(tf.concat([vsz,[1]],0))
            ##bug 
            dest_edges.append(tf.expand_dims(dedge,0))
            dest_shapes.append(dshape)

          if verbose:
            for k in range(depth):
               outvoxstr = ""
               if not tf.reduce_all(patch_shapes[k] == out_patch_shapes[k]):
                   outvoxstr = '  outvoxsize:' +  showten((patch_widths[k]/(out_patch_shapes[k]-1)),2)
               print("level "+str(k) + ":  shape:"+ showten(patch_shapes[k],0)+ "->"+ showten(out_patch_shapes[k],0)+ "  width(mm):"+showten(patch_widths[k],1) + "\n" +
                    '  voxsize:' +  showten((patch_widths[k]/(patch_shapes[k]-1)),2) +
                    '  (rel. to input:' +  showten((patch_widths[k]/(patch_shapes[k]-1))/(input_width/(tensor(input_shape)-1) ),2) + ')' +
                    " " + outvoxstr 
                     )
               print('  dest_shape:' + showten(dest_shapes[k],0))
            

          return { 'patch_widths' : patch_widths,
                  'patch_shapes' : patch_shapes,
                  'out_patch_shapes' : out_patch_shapes,
                  'dest_edges' : dest_edges,
                  'dest_shapes' : dest_shapes,
                  'depth' : self.depth,                  
                  }

      def toboxes(edges):
          i_ = [0,1,3] if self.ndim == 2 else [0,1,2,3]
          boxes = edges[i_,:]
          return tensor(boxes[:,i_])


      if isinstance(resolution_,dict) and "input_edges" in resolution_:
          src_boxes = toboxes(resolution_['input_edges'])
          src_voxsize = tf.math.sqrt(tf.reduce_sum(src_boxes[0:self.ndim,0:self.ndim]**2,0))
          src_width = tf.math.sqrt(tf.reduce_sum(src_boxes[0:self.ndim,0:self.ndim]**2,0))*(tensor(trainset_.shape[1:-1])-1)
          if "output_edges" in resolution_ and labels_ is not None:
              src_boxes_labels = toboxes(resolution_['output_edges'])
              src_width_labels = tf.math.sqrt(tf.reduce_sum(src_boxes_labels[0:self.ndim,0:self.ndim]**2,0))*(tensor(labels_.shape[1:-1])-1)                        
          else:                  
              src_boxes_labels = None
              src_width_labels = None          
      else:
          if isinstance(resolution_,dict):
               resolution_ = resolution_['voxsize']
          src_width =  (tensor(trainset_.shape[1:-1])-1)*tensor(resolution_)
          src_boxes = tf.linalg.diag(tensor(list(resolution_)+[1]))
          src_boxes_labels = None
          src_width_labels = None
          
      src_boxes = tf.tile(tf.expand_dims(src_boxes,0),[trainset_.shape[0],1,1])
      if src_boxes_labels is not None:
          src_boxes_labels = tf.tile(tf.expand_dims(src_boxes_labels,0),[trainset_.shape[0],1,1])

      patching_params = getPatchingParams(src_width,trainset_.shape[1:-1],src_boxes,resolution_,self.depth)


      # if isinstance(resolution_,dict) and "input_edges" in resolution_:          
      #     destboxes = patching_params['dest_edges']
      #     ##bug
      #     worldtrafo = tf.einsum('ik,k->ik',toboxes(resolution_['input_edges']),1.0/tf.concat([src_voxsize,[1]],0))
      #     for k in range(len(destboxes)):
      #         destboxes[k] = tf.einsum('ij,cjk->cik',worldtrafo,destboxes[k])
      
        
      
        
      aug_fac = lambda level: 1.0
      if augment is not None:                  
          if isinstance(augment,dict):
              if 'gamma' in augment:
                  aug_fac = lambda level: ((level+1)/self.depth)**augment['gamma']

     
    
      localCrop = lambda x,level : self.createCropsLocal(trainset_,
                         src_boxes,
                         src_width,                         
                         labels_,                         
                         src_boxes_labels,
                         src_width_labels,                         
                         x, level,
                         patching_params,
                         generate_type=generate_type,
                         snapper=snapper,
                         jitter = jitter,
                         overlap = overlap,
                         dphi1=dphi1*aug_fac(level),
                         dphi2=dphi2*aug_fac(level),
                         flip=flip,
                         dscale = dscale*aug_fac(level),
                         independent_augmentation=independent_augmentation,
                         pixel_noise = pixel_noise,
                         input_transform_behaviour = input_transform_behaviour,
                         label_transform_behaviour = label_transform_behaviour,                         
                         balance=balance,
                         num_patches=num_patches,
                         branch_factor=branch_factor,
                         training=training,
                         verbose=verbose)


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
                           
              if self.categorial_label is None:
                  if balance is not None and 'label_range' in balance:
                        label_range = tf.cast(balance['label_range'],dtype=tf.int32)
                        labs = tf.gather(labs,label_range,axis=self.ndim+1)           
                  #if balance is not None and 'label_reduce' in balance:
                  #    labs = tf.reduce_sum(labs,axis=-1,keepdims=True)
                  indicator = tf.math.reduce_max(labs,list(range(1,self.ndim+1)))
              else:
                  tmp = []
                  for j in self.categorial_label:
                      tmp.append(tf.reduce_max(tf.cast(labs==j,dtype=tf.float32),list(range(1,self.ndim+1))))
                  indicator = tf.concat(tmp,1)
                    
              indicator = tf.cast(indicator>0,dtype=tf.float32)                                
              cur_ratio = tf.expand_dims(tf.math.reduce_mean(indicator,axis=0),1)              
              balances[k].append(cur_ratio)
              if verbose:
                  print(' level: ' + str(k) + ' balance: ' + str(np.transpose(cur_ratio.numpy())[0]) )

        
      # if we want to train in tree mode we have to complete the tree
      if reptree:
        scales = self.tree_complete(scales)

      # # for repmatting the classlabels
      # if scales[k]['class_labels'] is not None and (self.model.classifier_train or self.model.spatial_max_train):
      #     for k in range(len(scales)):
      #         m = scales[k]['data_cropped'].shape[0] // scales[k]['class_labels'].shape[0]              
      #         tmp = scales[k]['class_labels']
      #         if len(tmp.shape) == 1:
      #             tmp = tf.expand_dims(tmp,1)
      #         tmp = tf.reshape(tf.tile(tf.expand_dims(tmp,1),[1,m,1]),[m*tmp.shape[0],tmp.shape[1]])              
      #         scales[k]['class_labels'] = tmp

        

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

    print("")

      
    if len(balances[0]) > 0:
        for k in range(self.depth):
            cur_ratio = tf.reduce_mean(tf.concat(balances[k],1),1)
            print(' level: ' + str(k) + ' avg. balance: ' + str(cur_ratio.numpy()) )
        

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
        elif smoothfac == 'boxcar' or smoothfac == 'max' or smoothfac == 'mixture'  or smoothfac == 'globalmax':
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
            if smoothfac == 'globalmax':
                conv_box = globalmax
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
      
      
      if interp_type == 'NN':      

           # for k in range(repfac):              
           #    tmp = tf.gather_nd(data_smoothed, parent_box_index[ds0*k:ds0*(k+1),...],batch_dims=1) 
           #    res_data.append(tmp)
           # res_data = tf.concat(res_data,0)


           _parent_box_index = []
         
           bdidx = tf.range(ds0,dtype=parent_box_index.dtype)
           for i in range(self.ndim+1):
               bdidx = tf.expand_dims(bdidx,1)
           bdidx = tf.tile(bdidx,[1] + parent_box_index.shape[1:-1]+ [1])
          
           for k in range(repfac):
              tmp = parent_box_index[ds0*k:ds0*(k+1),...]
              tmp = tf.concat([bdidx, tmp],self.ndim+1)
              _parent_box_index.append(tmp)
           _parent_box_index = tf.concat(_parent_box_index,0)
           res_data = tf.gather_nd(data_smoothed, _parent_box_index,batch_dims=0) 
      else:         
          for k in range(repfac):              
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
      
      

    
    
  def createCropsLocal(self,
                         src_data,            # the raw data img
                         src_boxes,           # affine matrix of raw data
                         src_width,           # width of raw data in mm
                         src_labels,          # the raw label img
                         src_boxes_labels,    # affine matrix of labels
                         src_width_labels,    # width of label array
                         crops, 
                         level,
                         patching_params,
                         generate_type='random',
                         snapper = None,
                         jitter = 0,
                         overlap = 0,
                         dphi1=0,
                         dphi2=0,
                         flip=None,                         
                         dscale = 0,
                         independent_augmentation = False,
                         pixel_noise = 0,
                         input_transform_behaviour = None,
                         label_transform_behaviour = None,
                         balance=None,
                         num_patches=1,
                         training=False,
                         branch_factor=1 ,
                         verbose=False):
        
        if balance is not None:
            balance = balance.copy()
            if 'label_range' in balance and balance['label_range'] is not None:
                balance['label_range'] = tf.cast(balance['label_range'],dtype=tf.int32)
            else:
                balance['label_range'] = tf.cast(range(self.model.num_labels),dtype=tf.int32)
            if 'label_weight' in balance  and balance['label_weight'] is not None:
                balance['label_weight'] = tf.cast(balance['label_weight'],dtype=src_data.dtype)
            else:
                balance['label_weight'] = tf.cast([1]*self.model.num_labels,dtype=src_data.dtype)
    
        
    
        tensor = lambda a : tf.cast(a,dtype=self.ftype)
        int32 = lambda a : tf.cast(a,dtype=tf.int32)
       
        src_bdim = src_data.shape[0]
        src_shape = tensor(src_data.shape[1:-1])
        if src_labels is not None:
            src_shape_labels = tensor(src_labels.shape[1:-1])
            
            

    
        patch_widths = patching_params['patch_widths']
        patch_shapes = patching_params['patch_shapes']
        out_patch_shapes = patching_params['out_patch_shapes']
        dest_edges = list(map(lambda x: tf.tile(x,[src_bdim,1,1]),patching_params['dest_edges']))
        dest_shapes = patching_params['dest_shapes']
        depth = patching_params['depth']
        
        
                
        def fdim_transform(res_data,orients,transform_behaviour):
            if transform_behaviour is None:
                return res_data
            else:
                T = transform_behaviour(orients)
                fac = 1/res_data.shape[-1]
                if nD==2:
                    return tf.einsum('bij,bxyj->bxyi',T,res_data)*fac
                else:
                    return tf.einsum('bij,bxyzj->bxyzi',T,res_data)*fac

        
        def compindex(dbox,lbox,dshape,lshape,noise,interptyp,offs):        
            e = tf.einsum('bxy,kbyz->kbxz',tf.linalg.inv(dbox,adjoint=False),lbox)
            box_index = grid(e,lshape,noise)-offs*0.5
            return clip(box_index,dshape,interptyp)
    
        
        def grid(edges,shape,pixel_noise):
            
            old_shape = edges.shape
            B = tf.reduce_prod(old_shape[0:-2])
            edges = tf.reshape(edges,[B,old_shape[-2],old_shape[-1]])
            
            ex1 = lambda x: tf.expand_dims(x,1)
        
            nD = len(shape)
            if nD == 2:
                A = tf.meshgrid(tf.range(0,shape[0],dtype=edges.dtype),tf.range(0,shape[1],dtype=edges.dtype),indexing='ij')
            if nD == 3:
                A = tf.meshgrid(tf.range(0,shape[0],dtype=edges.dtype),tf.range(0,shape[1],dtype=edges.dtype),tf.range(0,shape[2],dtype=edges.dtype),indexing='ij')
            for k in range(nD):
                A[k] = A[k] + tf.random.normal(A[k].shape)*pixel_noise
                A[k] = tf.expand_dims(tf.expand_dims(A[k],0),nD+1)
            R = tf.concat(A,nD+1)
            R = tf.tile(R, [edges.shape[0]] + [1]*(nD+1) )
            if nD == 2:
                R = tf.einsum('bxy,bijy->bijx',edges[...,0:nD,0:nD],R) 
                R = R+ex1(ex1(edges[:,0:nD,-1]))
            else:
                R = tf.einsum('bxy,bijky->bijkx',edges[...,0:nD,0:nD],R) 
                R = R+ex1(ex1(ex1(edges[:,0:nD,-1])))
            return R
    
        
        def clip(R,sz,interp_type):
            nD = len(sz)
            
            if level == 0:
                itype = tf.int64
            else:
                itype = tf.int32
            
            if interp_type == 'NN': # cast index to int
                R = tf.dtypes.cast(tf.floor(R+0.5),dtype=itype)            
                uplim = tf.cast(sz,itype)-1
            else:
                uplim = tf.cast(sz,itype)-2
            ex = lambda x: tf.expand_dims(x,0)
            uplim = ex(ex(ex(uplim)))
            if nD == 3:
                uplim = ex(uplim)
            
            R = tf.where(R<0,0,R)
            R = tf.where(R>uplim,uplim,R)
                            
            return R
        
        
        def quaternion(q):
            x = tf.expand_dims(q[...,0:1],2)
            y = tf.expand_dims(q[...,1:2],2)
            z = tf.expand_dims(q[...,2:3],2)
            r = tf.math.sqrt(x*x+y*y+z*z)
            rmod = r-tf.math.floor(r)
            sq = rmod/(r+0.00001)
            x = sq*x                   
            y = sq*y
            z = sq*z
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
            null = x*0
            return tf.concat([ tf.concat([Rxx,Rxy,Rxz,null],1),
                               tf.concat([Ryx,Ryy,Ryz,null],1),
                               tf.concat([Rzx,Rzy,Rzz,null],1),
                               tf.concat([null,null,null,1+null],1)],2)
        
        def draw_center_bylabel(label,balance,noisefac,nD,M):        
            
              if len(label.shape) < nD+2:
                  label = tf.expand_dims(label,3)
            
              ratio = balance['ratio']
              label_range = balance['label_range']
              label_weight = balance['label_weight']
              label_reduce = None
              if 'label_reduce' in balance:
                  label_reduce = balance['label_reduce']
              if 'label_weight' in balance and balance['label_weight'] is not None:
                  for k in range(nD):
                      label_weight = tf.expand_dims(label_weight,0)
                
              points_tot = []
              for k in range(label.shape[0]):
                  L = label[k,...]
                  if self.categorial_label is None:
                      if label_range is not None:
                          L = tf.gather(L,label_range,axis=nD)
                      if label_weight is not None:
                          L = L*label_weight
                  else:
                      tmp = 0;
                      for j in self.categorial_label:
                            tmp = tmp + tf.cast(L==j,dtype=tf.float32)
                      L = tmp
                  if label_reduce is not None:
                      L =tf.reduce_sum(L,axis=-1,keepdims=True)
                  L = np.amax(L,nD)
                  L = 1.0*(L>0)
                  sz = L.shape
                  cnt=0
            
                  numvx = np.prod(sz)
                  pos = tf.reduce_sum(L)
                  neg = numvx-pos
                  
                  if  pos == 0 or pos-numvx*ratio > -0.01:
                 #     print("warning: cannot achieve desired sample ratio, taking uniform")
                      P = L*0+1
                  else:
                      background_p = (1-ratio)*pos/(numvx*ratio-pos)
                      P = (L + background_p)
                      
                  p = tf.reshape(P,[numvx])
                  p = tf.cast(p,dtype=tf.float64)
                      
                  p = p/tf.reduce_sum(p)
                  idx = np.random.choice(numvx,M,p=p)
                  R = np.transpose(np.unravel_index(idx,sz))
                  R = R + np.random.normal(size=R.shape)*tf.expand_dims(noisefac,0)*0.2                  
                  R = R + np.random.uniform(low=0,high=1,size=R.shape)
                  points = R/sz
                  
                  points = tf.expand_dims(tf.cast(points,dtype=tf.float32),1)
                              
                  points_tot.append(points)
              
              centers = tf.concat(points_tot,1)            
                  
              return centers
        
         
        def draw_boxes(edges,shape,width,out_shape,out_width,label,N):
        
            def randU2d(dphi):
                phi = tf.random.normal([b*N,1,1])*dphi
                U = tf.concat([tf.concat([tf.math.cos(phi),-tf.math.sin(phi),phi*0],1),
                               tf.concat([tf.math.sin(phi),tf.math.cos(phi),phi*0],1),
                               tf.concat([0*phi,0*phi,1+phi*0],1)],               2)
                return U
        
            def randU3d(dphi):                
                phi = tf.random.normal([b*N,3])*dphi
                phi = phi / (tf.keras.backend.epsilon()+tf.math.sqrt(tf.reduce_sum(phi**2,1,keepdims=True)))
                phi = phi * tf.random.uniform([b*N,1])*tf.math.sin(tf.reduce_sum(dphi) )
                U = quaternion(phi)
                return U
                
            nD = len(shape)
            randrot = randU2d if nD==2 else randU3d
            b = edges.shape[0]
            
            
            ed = lambda a : tf.expand_dims(tf.expand_dims(a,0),0)
            w = (1-overlap)*out_width/width
        
            # rnd points inside patches
            if generate_type == "random" or generate_type == "random_deprec":
                if balance is not None:
                    points = draw_center_bylabel(label,balance,out_width/width*shape,nD,N)         
                else:
                    points = tf.random.uniform([N,b,nD],minval=0,maxval=1,dtype=edges.dtype)
#                    points = tf.random.uniform([N,b,nD],minval=-1,maxval=1,dtype=edges.dtype)
 #                   points = 0.5*(1+tf.math.sign(points) * tf.math.pow(1-tf.math.abs(points),2))                    
                points = points*(shape-1)
                                                
            elif generate_type == "tree":
                # tree 
                nboxes = tf.math.floor(1/w + 1)
                ran = lambda i: tf.range(0.5,nboxes[i],dtype=edges.dtype)
                if nD == 2:
                     A = tf.meshgrid(ran(0),ran(1),indexing='ij')
                else:
                     A = tf.meshgrid(ran(0),ran(1),ran(2),indexing='ij')
                N = int32(tf.reduce_prod(nboxes))
                for k in range(nD):
                    A[k] = tensor(tf.reshape(A[k],[N,1,1]))
                    A[k] = A[k] + tf.random.uniform([N,b,1],minval=-jitter,maxval=jitter,dtype=edges.dtype)
                    A[k] = A[k]/nboxes[k]*(shape[k]-1)
                points = tf.concat(A,2)
                N = points.shape[0]
            else:
                assert False, "not a valid generate_type"
    
            # snap overlapping patches on edges
            if snapper[level] == 1:
                snap = ed((shape-1)*out_width/width*0.5)
                points = tf.where(points < snap*1, snap,points)
                points = tf.where(points > ed(shape-1)-snap*1, ed(shape-1)-snap,points)
            
            points = tf.einsum('bxy,Nby->Nbx',edges[...,0:nD,0:nD],points) + tf.expand_dims(edges[...,0:nD,-1],0)
            points = tf.reshape(points,[b*N,nD])
            
            
            if affine_augment is None or independent_augmentation:                        
                # rnd transformations
                R1 = randrot(dphi1)
                R2 = randrot(dphi2)
                S = 1+tf.random.normal([b*N,nD])*dscale
                S = S*(2*tensor(tf.random.uniform([b*N,nD]) > flip*0.5)-1)
                
                A = tf.linalg.diag(tf.concat([S,tf.ones([b*N,1])],1))
                A = tf.einsum('cxy,cyz->cxz',A,R1)
                A = tf.einsum('cxy,cyz->cxz',R2,A)
                R = tf.einsum('cxy,cyz->cxz',R2,R1)
            else:
                A = tf.tile(affine_augment,[N,1,1])
                R = tf.tile(rot_augment,[N,1,1])
            
            vxsz = tf.expand_dims(tf.expand_dims(tf.concat([out_width/(out_shape-1),[1]],0),0),1) 

          #  U = A * vxsz
            ##bug
            src_vxsz = tf.concat([tf.math.sqrt(tf.reduce_sum(src_boxes[0,:,0:nD]**2,0)),[1]],0)            
            src_vxsz = tf.expand_dims(tf.expand_dims(src_vxsz,0),0)
            U = tf.matmul(A,src_boxes/src_vxsz*vxsz)
            
            
            # assemble homogenous coords
            offs = tf.concat([points,tf.ones([b*N,1])],1) - tf.einsum('bxy,y->bx',U,tf.concat([(out_shape-1)/2,[1]],0))
            E = U+tf.concat([tf.zeros([b*N,nD+1,nD]),tf.expand_dims(offs,2)],2)
            E = tf.reshape(E,[N,b,nD+1,nD+1])
                    
            return E,R,A
        
    
        if level == 0:        
            last_boxes = src_boxes
            last_width = src_width
            last_shape = src_shape
            last_label = src_labels
            affine_augment = None
            rot_augment = None
            N=num_patches
        else:
            last_boxes = crops['local_boxes']        
            last_width = patch_widths[level-1]
            last_shape = patch_shapes[level-1]
            last_label = crops['labels_cropped']
            affine_augment = crops['affine_augment']
            rot_augment = crops['rot_augment']
            N=branch_factor
            
        nD = len(last_width)
        
    
        if flip is None:
            flip = tensor([0]*nD)
                        
        start = timer()
          
        local_boxes, rot_augment, affine_augment = draw_boxes(
                                 last_boxes, last_shape, last_width, 
                                 patch_shapes[level], patch_widths[level], 
                                 last_label,
                                 N)
        
    
        # the index which crops the subpatch out of the last patch
        local_box_index = compindex(last_boxes,local_boxes,last_shape,patch_shapes[level],0,self.interp_type,0)
    

        local_boxes = tf.reshape(local_boxes,[
                        tf.reduce_prod(local_boxes.shape[0:2])/src_bdim, src_bdim ,
                        nD+1,nD+1])  
        
        # the index which crops data the data out of the overall parent
        parent_box_index = compindex(src_boxes,local_boxes,src_shape,patch_shapes[level],
                                     pixel_noise,self.interp_type,0)

        if src_labels is not None:             
            parent_box_index_label = parent_box_index
            if src_boxes_labels is not None:
                lbox,lshape = src_boxes_labels,src_shape_labels
            else:                
                lbox,lshape = src_boxes,src_shape
            if not tf.reduce_all(patch_shapes[level] == out_patch_shapes[level]):
                relwid = 1/tf.concat([out_patch_shapes[level]/patch_shapes[level],[1]],0)
                local_boxes_labels = tf.einsum('Nbxy,y->Nbxy',local_boxes,relwid)
                parent_box_index_label = compindex(lbox,local_boxes_labels,lshape,out_patch_shapes[level],
                                                      pixel_noise,self.interp_type,0)
            elif src_boxes_labels is not None:
                parent_box_index_label = compindex(lbox,local_boxes,lshape,patch_shapes[level],
                                     pixel_noise,self.interp_type,0)
                


        # the index which scatters output into the global image        
        if dest_shapes[level] is not None and not training:
            vratio = tf.concat([patch_shapes[level]/out_patch_shapes[level],[1]],0)
            local_boxes_out = tf.einsum('Nbxy,y->Nbxy',local_boxes,vratio)            
            parent_box_scatter_index = compindex(dest_edges[level],local_boxes_out,dest_shapes[level],out_patch_shapes[level],
                                                 0,self.scatter_type,0)
        else:
            parent_box_scatter_index = None       
            
        local_boxes = tf.reshape(local_boxes,[tf.reduce_prod(local_boxes.shape[0:2]) ,nD+1,nD+1])
    
        
        relres = [1]*nD
            
        if verbose:
          print("--------- cropping, level ",level)
          
            
        ############## do the actual cropping
        res_data = self.crop(src_data,parent_box_index,relres,self.get_smoothing(level,'data'),interp_type=self.interp_type,verbose=verbose)       
        res_data = fdim_transform(res_data,rot_augment,input_transform_behaviour)
        if src_labels is not None:                  
          res_labels = self.crop(src_labels,parent_box_index_label,relres,self.get_smoothing(level,'label'),interp_type=self.interp_type,verbose=verbose)
          res_labels = fdim_transform(res_labels,rot_augment,label_transform_behaviour)
        else:
            res_labels = None

            
        if verbose:
           end = timer()
           print(" #patches:" + str(res_data.shape[0]) + " time/patch:" + ("{:.3f}ms").format(1000*((end - start)/(res_data.shape[0]))) )
           print(" elapsed: " + ("{:.3f}ms").format(1000*(end - start)) )

        
        if training:
            parent_box_index = None
    
    
    
        return {"data_cropped" : res_data, 
                "labels_cropped" : res_labels, 
                               
                  # these are crop coordinates refering to the last upper scale
                  "local_boxes" : local_boxes, 
                  "local_box_index": local_box_index,
    
                  # these are crop coordinates refereing to the very original image
                  "parent_box_index": parent_box_index,
                  "parent_box_scatter_index": parent_box_scatter_index,
                  "dest_full_size":dest_shapes[level],
                  "affine_augment":affine_augment,
                  "rot_augment":rot_augment,
    
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
                 #'parent_box_scatter_index',
                 'affine_augment',
                 'rot_augment',
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






