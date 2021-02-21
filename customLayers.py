#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 11:37:27 2020

@author: reisertm
"""

from tensorflow.keras import layers
import tensorflow as tf
import numpy as np

#import patchwork



custom_layers = {}


# https://stackoverflow.com/questions/61649083/tf-keras-how-to-reuse-resnet-layers-as-customize-layer-in-customize-model

def createUnet_v1(depth=4,outK=1,multiplicity=1,feature_dim=5,nD=3,
                  padding='SAME',centralDense=None,noBridge=False,verbose=False,input_shape=None):

  if nD == 3:
      _conv = layers.Conv3D
      _convT = lambda *args, **kwargs: layers.Conv3DTranspose(*args, **kwargs,strides=(2,2,2))
      _maxpool = lambda: layers.MaxPooling3D(pool_size=(2,2,2))
  elif nD == 2:
      _conv = layers.Conv2D
      _convT = lambda *args, **kwargs: layers.Conv2DTranspose(*args, **kwargs,strides=(2,2))
      _maxpool = lambda: layers.MaxPooling2D(pool_size=(2,2))
  
  def BNrelu():
      return [layers.BatchNormalization(), layers.LeakyReLU()]
  
  if padding == 'VALID':
      def conv_down(fdim):
           return _conv(fdim,3,padding='VALID') 
      def conv_up(fdim,even):
           return _convT(fdim,4+even,padding='VALID' )
      def conv(outK):
           return _conv(outK,4,padding='SAME') 
      offs = [0,1,0,0,0,0,0]
  else:
      def conv_down(fdim):
           return _conv(fdim,3,padding='SAME') 
      def conv_up(fdim,even):
           return _convT(fdim,3,padding='SAME' )
      def conv(outK):
           return _conv(outK,3,padding='SAME') 
      offs = [0,0,0,0,0,0]
      
  if not isinstance(feature_dim,list):
      fdims=[]
      for z in range(depth):
          fdims.append(feature_dim*(1+z))     
  else:
      fdims = feature_dim


  theLayers = {}
  for z in range(depth):
      
    if input_shape is not None:
        for r in range(len(input_shape)):
            input_shape[r] = input_shape[r]/2
      
    fdim = fdims[z]
    id_d = str(1000 + z+1)
    id_u = str(2000 + depth-z+1)
    if noBridge:
        theLayers[id_d+"conv0"] =  [conv_down(fdim) ]+BNrelu()
    else:
        theLayers[id_d+"conv0"] = [{'f': [conv_down(fdim)]+BNrelu() } , {'f': conv(fdim), 'dest':id_u+"relu" }  ]
                
    for k in range(multiplicity-1):
        theLayers[id_d+"conv"+str(k+1)] = [conv(fdim)] + BNrelu()
    theLayers[id_d+"relu"] = _maxpool()
     
    if centralDense is not None:
        if z == depth-1:
            theLayers[id_d+"s_central_0"] = layers.Flatten()
            for k in range(len(centralDense['feature_dim'])):
                theLayers[id_d+"s_central_1_"+str(k)] = [layers.Dense(centralDense['feature_dim'][k])]+BNrelu()
            theLayers[id_d+"s_central_2"] = 'reshape_' + str(centralDense['out'])
    
    
    
    theLayers[id_u+"conv0"] = conv_up(fdim,offs[z])
    for k in range(multiplicity-1):
        if k == multiplicity-1:
            theLayers[id_u+"conv"+str(k+1)] = [conv(fdim)] 
        else:
            theLayers[id_u+"conv"+str(k+1)] = [conv(fdim)] + BNrelu()
            
    theLayers[id_u+"relu"] = BNrelu()
  theLayers["3000"] =  [layers.Dropout(rate=0.5), conv(outK)]
  #return patchwork.CNNblock(theLayers,verbose=verbose)
  return CNNblock(theLayers,verbose=verbose)



def createUnet_v2(depth=4,outK=1,multiplicity=1,feature_dim=5,nD=3,dropout=False,
                  padding='SAME',centralDense=None,noBridge=False,verbose=False,input_shape=None):
  if nD == 3:
      strides = [2,2,2]
      _conv = layers.Conv3D
      _convT = lambda *args, **kwargs: layers.Conv3DTranspose(*args, **kwargs,strides=strides)
      _maxpool = lambda: layers.MaxPooling3D(pool_size=strides)
  elif nD == 2:
      strides = [2,2]
      _conv = layers.Conv2D
      _convT = lambda *args, **kwargs: layers.Conv2DTranspose(*args, **kwargs,strides=strides)
      _maxpool = lambda: layers.MaxPooling2D(pool_size=(2,2))
  
  def BNrelu():
      return [layers.BatchNormalization(), layers.LeakyReLU()]
  
  if padding == 'VALID':
      def conv_down(fdim):
           return _conv(fdim,3,padding='VALID') 
      def conv_up(fdim,even):
           return _convT(fdim,4+even,padding='VALID' )
      def conv(outK):
           return _conv(outK,4,padding='SAME') 
      offs = [0,1,0,0,0,0,0]
  else:
      def conv_down(fdim):
           return _conv(fdim,3,padding='SAME') 
      def conv_up(fdim,even):
           return _convT(fdim,3,padding='SAME' )
      def conv(outK):
           return _conv(outK,3,padding='SAME') 
      offs = [0,0,0,0,0,0]
      
  if not isinstance(feature_dim,list):
      fdims=[]
      for z in range(depth):
          fdims.append(feature_dim*(1+z))     
  else:
      fdims = feature_dim

  if input_shape is not None:
     tmp = input_shape
     input_shape = []
     input_shape.extend(tmp)
      
      
  theLayers = {}
  for z in range(depth):
      
      
    fdim = fdims[z]
    id_d = str(1000 + z+1)
    id_u = str(2000 + depth-z+1)
    if noBridge:
        theLayers[id_d+"0_conv"] =  [conv_down(fdim) ]+BNrelu()
    else:
        theLayers[id_d+"0_conv"] = [{'f': [conv_down(fdim)]+BNrelu() } , {'f': conv(fdim), 'dest':id_u+"5_relu" }  ]
                
    for k in range(multiplicity-1):
        theLayers[id_d+"1_conv"+str(k+1)] = [conv(fdim)] + BNrelu()
    theLayers[id_d+"2_relu"] = [{'f':_maxpool() } , {'f':identity(), 'maxadd':id_u+"5_relu"}]
#    theLayers[id_d+"2_relu"] = _maxpool() 
     
    if centralDense is not None:
        if z == depth-1:
            theLayers[id_d+"3_central_0"] = layers.Flatten()
            for k in range(len(centralDense['feature_dim'])):
                theLayers[id_d+"3_central_1_"+str(k)] = [layers.Dense(centralDense['feature_dim'][k])]+BNrelu()
            theLayers[id_d+"3_central_2"] = 'reshape_' + str(centralDense['out'])
    
    
    
    theLayers[id_u+"4_conv"] = conv_up(fdim,offs[z])
    for k in range(multiplicity-1):
        if k == multiplicity-1:
            theLayers[id_u+"4_conv"+str(k+1)] = [conv(fdim)] 
        else:
            theLayers[id_u+"4_conv"+str(k+1)] = [conv(fdim)] + BNrelu()
            
    theLayers[id_u+"5_relu"] = BNrelu()
    
    if input_shape is not None:
        for r in range(len(input_shape)):
            input_shape[r] = input_shape[r]//strides[r]
            if input_shape[r] == 1:
                strides[r] = 1
                
    
  if dropout:
      theLayers["9_final"] =  [layers.Dropout(rate=0.5), conv(outK)]
  else:
      theLayers["9_final"] =  conv(outK)
      
  #return patchwork.CNNblock(theLayers,verbose=verbose)
  return CNNblock(theLayers,verbose=verbose)



def createUnet_bi(depth=4,outK=1,multiplicity=1,feature_dim=5,nD=3,verbose=False):

          

  if nD == 3:
      _conv = lambda *args, **kwargs: biConvolution(*args, **kwargs, nD=3)
      _convT = lambda *args, **kwargs: biConvolution(*args, **kwargs, nD=3,transpose=True,strides=(2,2,2))
      _maxpool = lambda: layers.MaxPooling3D(pool_size=(2,2,2))
  elif nD == 2:
      _conv = lambda *args, **kwargs: biConvolution(*args, **kwargs, nD=2)
      _convT = lambda *args, **kwargs: biConvolution(*args, **kwargs, nD=2,transpose=True,strides=(2,2))
      _maxpool = lambda: layers.MaxPooling2D(pool_size=(2,2))
  
  def BNrelu():
      return [layers.BatchNormalization(), layers.LeakyReLU()]
  
  def conv_down(fdim):
         return _conv(out_n=fdim) 
  def conv_up(fdim,even):
         return _convT(out_n=fdim)
  def conv(outK):
         return _conv(out_n=outK) 
  offs = [0,0,0,0,0,0]
      
  if not isinstance(feature_dim,list):
      fdims=[]
      for z in range(depth):
          fdims.append(feature_dim*(1+z))     
  else:
      fdims = feature_dim

  theLayers = {}
  for z in range(depth):
    fdim = fdims[z]
    id_d = str(1000 + z+1)
    id_u = str(2000 + depth-z+1)
    theLayers[id_d+"conv0"] = [{'f': [conv_down(fdim)]+BNrelu() } , {'f': conv(fdim), 'dest':id_u+"relu" }  ]
    for k in range(multiplicity-1):
        theLayers[id_d+"conv"+str(k+1)] = [conv(fdim)] + BNrelu()
    theLayers[id_d+"relu"] = _maxpool()
    theLayers[id_u+"conv0"] = conv_up(fdim,offs[z])
    for k in range(multiplicity-1):
        if k == multiplicity-1:
            theLayers[id_u+"conv"+str(k+1)] = [conv(fdim)] 
        else:
            theLayers[id_u+"conv"+str(k+1)] = [conv(fdim)] + BNrelu()
            
    theLayers[id_u+"relu"] = BNrelu()
  theLayers["3000"] =  [layers.Dropout(rate=0.5), conv(outK)]
 # return patchwork.CNNblock(theLayers,verbose=verbose)
  return CNNblock(theLayers,verbose=verbose)




def simpleClassifier(depth=6,feature_dim=5,nD=2,outK=2,multiplicity=2,
                     verbose=False,activation='sigmoid'):

    
    
    if not isinstance(feature_dim,list):
          fdims=[]
          for z in range(depth):
              fdims.append(feature_dim*(1+z))     
    else:
          fdims = feature_dim


    
    def BNrelu(name=None):
      return [layers.BatchNormalization(name=name), layers.LeakyReLU()]
    
    if nD == 2:
        _conv = lambda *args, **kwargs: biConvolution(*args, **kwargs, nD=2)
        pooly = lambda: layers.MaxPooling2D(pool_size=(2,2))
    if nD == 3:
        _conv = lambda *args, **kwargs: biConvolution(*args, **kwargs, nD=3)
        pooly = lambda: layers.MaxPooling3D(pool_size=(2,2,2))
    
    
    theLayers = {}
    for z in range(depth):
      id_d = str(1000 + z+1)
      for i in range(multiplicity):
          theLayers[id_d+"_" + str(i) + "conv"] = _conv(out_n=fdims[z])
          theLayers[id_d+"_" + str(i) + "relu"] = BNrelu() 
      theLayers[id_d+"spool"] = [pooly()]
    theLayers["3001"] =  layers.Flatten()
    theLayers["3002"] =  layers.Dropout(rate=0.5)
    theLayers["3003"] =  layers.Dense(outK)
    if activation is not None:
        theLayers["3005"] = layers.Activation(activation)
      
    return CNNblock(theLayers,verbose=verbose)
    #return patchwork.CNNblock(theLayers,verbose=verbose)
    
    


def createTnet(nD=3, depth=2,fdims=None,out=1,ind=3,noise=0.5,ksize=3,padding='SAME',direction=True,verbose=False,input_shape=None):

  if nD == 3:
      _convS = lambda *args, **kwargs: layers.Conv3D(*args, **kwargs,strides=(1,1,1))
      _conv = lambda *args, **kwargs: layers.Conv3D(*args, **kwargs,strides=(2,2,2))
      _convT = lambda *args, **kwargs: layers.Conv3DTranspose(*args, **kwargs,strides=(2,2,2))
  elif nD == 2:
      _conv = lambda *args, **kwargs: layers.Conv2D(*args, **kwargs,strides=(2,2,2))
      _convT = lambda *args, **kwargs: layers.Conv2DTranspose(*args, **kwargs,strides=(2,2))
  
  def BNrelu():
      return [layers.BatchNormalization(), layers.LeakyReLU()]
  
  def conv_down(fdim):
         return _conv(fdim,ksize,padding='SAME') 
  def conv_up(fdim):
         return _convT(fdim,ksize,padding='SAME' )
  def conv(fdim):
         return _convS(fdim,ksize,padding='SAME' )
  
  n = ind
  #%%
  if fdims is None:
      fdims = [n]
      fac = np.power(out/n,1/(depth))
      for k in range(depth):
          fdims.append(fdims[-1]*fac)
      fdims = list(map(lambda x: np.int32(np.floor(x)),fdims))
      if not stayUp:  
          fdims[-1] = out
      
  #%%    

  theLayers = {}
  if direction == 0:
      for z in range(depth):            
        fdim = fdims[z]
        id_u = str(1000 + z+1)
        
        theLayers[id_u+"conv0"] = [conv_up(fdims[z+1]) ]+BNrelu()
        if z == depth-1:
            theLayers[id_u+"conv1"] = conv(out)
  else:        
      for z in range(depth):            
        fdim = fdims[z]
        id_u = str(1000 + z+1)
        
        theLayers[id_u+"conv0"] = [conv_down(fdims[z+1]) ]+BNrelu()
        if z == depth-1:
            theLayers[id_u+"conv1"] = conv(out)
    
            
  return CNNblock(theLayers,verbose=verbose)


custom_layers['createTnet'] = createTnet



class identity(layers.Layer):

  def __init__(self,**kwargs):
    super().__init__(**kwargs)
  def call(self, image):         
      return image
  
custom_layers['identity'] = identity




class Scramble(layers.Layer):

  def __init__(self,nD,noise=0.5,**kwargs):
    super().__init__(**kwargs)
    self.nD = nD
    self.noise = noise
    
    
  def get_config(self):
        config = super().get_config().copy()
        config.update(
        {
            'noise': self.noise,
            'nD': self.nD
        } )    
        return config                  
    
    
  def call(self, image, training=False):    
      
    #if training == False:
    #    return image
        
      
    def grid(bdim,shape,pixel_noise):
        ex1 = lambda x: tf.expand_dims(x,0)    
        nD = self.nD
        if nD == 2:
            A = tf.meshgrid(tf.range(0,shape[0],dtype=tf.float32),tf.range(0,shape[1],dtype=tf.float32),indexing='ij')
        if nD == 3:
            A = tf.meshgrid(tf.range(0,shape[0],dtype=tf.float32),tf.range(0,shape[1],dtype=tf.float32),tf.range(0,shape[2],dtype=tf.float32),indexing='ij')
        for k in range(nD):
            q = tf.expand_dims(tf.expand_dims(A[k],0),nD+1)
            noise = tf.expand_dims(tf.random.normal([bdim] + A[k].shape),(nD+1))
            A[k] = q + noise*pixel_noise

        A = tf.concat(A,nD+1)
        A = tf.cast(tf.math.floor(A+0.4999),dtype=tf.int32)
        A = tf.where(A<0,0,A)
        
        s = tf.cast(shape[0:nD],dtype=tf.int32)-1
        for k in range(nD+1):
            s = tf.expand_dims(s,0)        
        A = tf.where(A>s,s,A)
        
        
        return A
    
    shape = image.shape
    g = grid(shape[0],shape[1:],self.noise)
    return tf.gather_nd(image,g,batch_dims=1)

    
custom_layers['Scramble'] = Scramble



class HistoMaker(layers.Layer):

  def __init__(self,nD=2,out=4,scfac=0.2,scaling=1.0,init='ct',normalize=True,trainable=False,dropout=False,**kwargs):
    super().__init__(**kwargs)

    self.init = init
    self.scfac = scfac
    self.nD = nD
    self.normalize = normalize
    self.trainable = trainable
    self.dropout = dropout
    self.scaling = scaling
    
    self.dropout_layer = None

    if dropout is not None or dropout >0:
        self.dropout_layer = layers.Dropout(rate=dropout)
        self.normalize = False
    
    if init is None:
        if nD == 2:
            self.conv = layers.Conv2D(out,1)
        else:
            self.conv = layers.Conv3D(out,1)
    else:
        if isinstance(init,str):
            if init == 'ct':        
                self.centers = np.array([-1000,-500,-100,-50,-25,0,25,50,100,500,1000])
            else:
                assert False, "init not defined for histolayer"
        else:
            self.centers = np.array(init)
        width = np.convolve(self.centers,[-1,0,1],mode='valid')
        self.width = np.abs(np.append(np.append([width[0]],width),[width[-1]]))*scfac
        self.centers = -self.centers/self.width
        out = len(self.centers)

        def bias_initializer(shape,dtype):
            bias = tf.cast(self.centers,dtype=dtype)
            bias = tf.reshape(bias,shape)
            return bias
    
        def kernel_initializer(shape,dtype):
            sc = 1/tf.cast(self.width,dtype=dtype)
            sc = tf.reshape(sc,shape)
            return sc

        if nD == 2:
            self.conv = layers.Conv2D(out,1,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer)
        else:
            self.conv = layers.Conv3D(out,1,kernel_initializer=kernel_initializer,bias_initializer=bias_initializer)
    
    self.out = out
    self.conv.trainable = trainable

    
    
  def get_config(self):
        config = super().get_config().copy()
        config.update(
        {
            'scfac': self.scfac,
            'scaling': self.scaling,
            'normalize': self.normalize,            
            'init': self.init,
            'out': self.out,
            'nD': self.nD,
            'dropout':self.dropout
        } )    
        return config                  
      
  def call(self, image):    
      x = image*self.scaling
      x = self.conv(x)
      ch = 1/tf.math.cosh(x)
      if self.dropout_layer is not None:
          ch = self.dropout_layer(ch)
      if self.normalize:
          return ch / tf.reduce_sum(ch,axis=-1,keepdims=True)
      else:
          return ch

custom_layers['HistoMaker'] = HistoMaker
  

class sigmoid_softmax(layers.Layer):

  def __init__(self,**kwargs):
    super().__init__(**kwargs)
  def call(self, image):         
      e = tf.math.exp(image)
      s = (1+tf.reduce_sum(e,axis=-1,keepdims=True))
      return  e/s
  
custom_layers['sigmoid_softmax'] = sigmoid_softmax


class ct_preproc(layers.Layer):

  def __init__(self,**kwargs):
    super().__init__(**kwargs)
  def call(self, image):       
      x = tf.math.log(tf.math.maximum(image+200,0.001))
      return tf.math.maximum(x-4,0)

    
custom_layers['ct_preproc'] = ct_preproc



class normalizedConvolution(layers.Layer):

  def __init__(self, nD=3, out_n0=7,  ksize0=3, 
                     out_n1=3,  ksize1=9, eps=0.001,**kwargs):
    super().__init__(**kwargs)
    if nD == 2:
        self.conv0 = layers.Conv2D(out_n0,ksize0,use_bias=False,padding='SAME') 
        self.conv1 = layers.Conv2D(out_n1,ksize1,use_bias=False,padding='SAME') 
    else:
        self.conv0 = layers.Conv3D(out_n0,ksize0,use_bias=False,padding='SAME') 
        self.conv1 = layers.Conv3D(out_n1,ksize1,use_bias=False,padding='SAME') 
    self.nD = nD
    self.out_n0=out_n0
    self.ksize0=ksize0
    self.out_n1=out_n1
    self.ksize1=ksize1
    self.eps = eps
    
  def get_config(self):
        config = super().get_config().copy()
        config.update(
        {
            'out_n0': self.out_n0,
            'ksize0': self.ksize0,
            'out_n1': self.out_n1,
            'ksize1': self.ksize1,
            'nD': self.nD,
            'eps': self.eps,
        } )    
        return config                  
  def call(self, image):
      x = self.conv0(image)
      y = self.conv1(image)
      n = tf.reduce_sum(y*y,axis=self.nD+1,keepdims=True)
      n = tf.math.sqrt(n+self.eps)
      x = x / n
      return x
      
custom_layers['normalizedConvolution'] = normalizedConvolution

############################ biConv


class biConvolution(layers.Layer):

  def __init__(self, out_n=7, ksize=3, padding='SAME',transpose=False,nD=2,strides=None,bias=False,**kwargs):
      
      
    super().__init__(**kwargs)
    self.out_n = out_n
    self.ksize = ksize
    self.padding = padding
    self.transpose = transpose
    self.nD = nD
    self.initializer = tf.random_normal_initializer(0, 0.05)
    self.num_alpha = 0
    self.isBi=True
    self.bias = bias
    
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
              weight_shape = (self.num_alpha+1,self.ksize,self.ksize,self.M,self.N)                        
          else:
              weight_shape = (self.num_alpha+1,self.ksize,self.ksize,self.ksize,self.M,self.N)
             
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

        offs = 0
        x = 0
        if self.bias:
            offs = 1
            kernel = self.weight[0,...]
            x = conv(image, kernel, strides=self.strides, padding=self.padding)
        for k in range(self.num_alpha-offs):
            kernel = self.weight[k+offs,...]
            alpha = alphas[:,k+offs:k+offs+1]
            for j in range(self.nD):
                alpha = tf.expand_dims(alpha,j+2)
            c = conv(image, kernel, strides=self.strides, padding=self.padding)
            x = x+alpha*c
    
    return x

custom_layers['biConvolution'] = biConvolution





#%%###############################################################################################

## A CNN wrapper to allow easy layer references and stacking

class CNNblock(layers.Layer):
  def __init__(self,theLayers=None,name=None,verbose=False):
    super().__init__(name=name)
    
    self.verbose = verbose
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
                
        
        if len(x.shape) == nD+2:
            self.last_spatial_shape = x.shape        
        if self.verbose:
            print("  " + f.name , "input_shape: " ,  x.shape)
        if hasattr(f,'isBi') and f.isBi:
            if self.verbose:
                if alphas is not None:
                    print("  " + f.name , "(bi) input_shape: " ,  alphas.shape)
            return f(x,alphas,training=training)
        else:
            return f(x,training=training)
        
    
    x = inputs
    nD = len(inputs.shape)-2 
    cats = {}
    applyout = []
    maxadds = {}
    for l in self.theLayers:
        cats[l] = []
        maxadds[l] = []

    if self.verbose:
        print("----------------------")
        if alphas is not None:
            print("with alphas: " , alphas.shape)

    for l in sorted(self.theLayers):
      
      if self.verbose:
          print("metalayer: " + l)
        
      # this gathers all inputs that might have been forwarded from other layers
      for r in maxadds[l]:
          x = x+r
      for r in cats[l]:
        if x is None:
          x = r
        else:
          x = tf.concat([x,r],nD+1)


      if isinstance(self.theLayers[l],str) and self.theLayers[l][0:7]  == 'reshape':
          prodsz = np.prod(self.last_spatial_shape[1:-1])
          outfdim = int(self.theLayers[l][8:])
          self.theLayers[l] = [layers.Dense(prodsz*outfdim),
                               layers.Reshape(self.last_spatial_shape[1:-1] + [outfdim])]


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
              if self.verbose:
                  print("          dest:"+dest)
              cats[dest].append(res)  
            if 'applyout' in d:
              dest = d['applyout']
              if self.verbose:
                  print("          applyout:"+dest)
              applyout.append(res)  
            if 'maxadd' in d:
              if self.verbose:
                  print("          dest:"+d['maxadd'])
              themaxs = tf.reduce_max(res,axis=list(range(1,nD+1)),keepdims=True)
              maxadds[d['maxadd']].append(themaxs)  
            
            if 'dest' not in d and 'maxadd' not in d and 'applyout' not in d:
              y = res
              
              
          x = y
        else:
          for f in a:
            #x = f(x,training=training)
            x = apply_fun(f,x)
      else:
        #x = a(x,training=training)
        x = apply_fun(a,x)

    if not training and len(applyout) > 0:
        return tf.concat(applyout,nD+1)
    else:
        return x



def createCNNBlockFromObj(obj,custom_objects=None):

  def tolayer(x):
      if isinstance(x,dict):
          if 'keras_layer' in x:
              return layers.deserialize({'class_name':x['name'],'config':x['keras_layer']},
                                        custom_objects=custom_objects)
          if 'CNNblock' in x:
              return createCNNBlockFromObj(x['CNNblock'],custom_objects=custom_objects)            
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
  if isinstance(theLayers,list):
      res = []
      for l in theLayers:
          res.append(createCNNBlockFromObj(l,custom_objects=custom_objects))
      return res
  else:
      return CNNblock(theLayers)
      








