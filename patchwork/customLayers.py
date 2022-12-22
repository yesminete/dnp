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
    if noBridge==2 and z==depth-1:
        theLayers[id_d+"conv0"] =  [conv_down(fdim) ]+BNrelu()
    elif noBridge==1:
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



def createUnet_v2(depth=4,outK=1,multiplicity=1,feature_dim=5,nD=3,dropout=False,nonlin=None,
                  padding='SAME',centralDense=None,noBridge=False,upsample2=False,verbose=False,input_shape=None):
  if nD == 3:
      strides = [2,2,2]
      if input_shape is not None:
          strides = tf.minimum(input_shape,strides).numpy()
      _conv = layers.Conv3D
      _convT = lambda *args, **kwargs: layers.Conv3DTranspose(*args, **kwargs,strides=strides)
      _maxpool = lambda: layers.MaxPooling3D(pool_size=strides)
  elif nD == 2:
      strides = [2,2]
      _conv = layers.Conv2D
      _convT = lambda *args, **kwargs: layers.Conv2DTranspose(*args, **kwargs,strides=strides)
      _maxpool = lambda: layers.MaxPooling2D(pool_size=(2,2))
    
  
  def BNrelu():
      if nonlin is None:
          return [layers.BatchNormalization(), layers.LeakyReLU()]
      elif nonlin == 'binaryLookup':
          return [layers.BatchNormalization(), binaryLookup()]
      else:
          return [layers.BatchNormalization(), layers.Activation(nonlin)]
  
  if padding == 'VALID':
      def conv_down(fdim):
           return _conv(fdim,3,padding='VALID') 
      def conv_up(fdim,even):
           return _convT(fdim,4+even,padding='VALID' )
      def conv(outK):
           return _conv(outK,4,padding='SAME') 
      offs = [0,1,0,0,0,0,0,0,0,0]
  else:
      def conv_down(fdim):
           return _conv(fdim,3,padding='SAME') 
      def conv_up(fdim,even):
           return _convT(fdim,3,padding='SAME' )
      def conv(outK):
           return _conv(outK,3,padding='SAME') 
      offs = [0,0,0,0,0,0,0,0,0,0]
      
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
    if noBridge==2 and z==depth-1:
        theLayers[id_d+"0_conv"] =  [conv_down(fdim) ]+BNrelu()    
    elif noBridge == 1:
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
                
    
  if dropout and dropout < 1.0: 
      if nD == 3:
          theLayers["9_final"] =  [layers.SpatialDropout3D(rate=float(dropout)), conv(outK)]
      else:
          theLayers["9_final"] =  [layers.SpatialDropout2D(rate=float(dropout)), conv(outK)]
  else:
      if upsample2:
          strides = [2]*nD
          theLayers["9_final2"] =  conv_up(outK,0)
      else:
          theLayers["9_final"] =  conv(outK)
      
  #return patchwork.CNNblock(theLayers,verbose=verbose)
  return CNNblock(theLayers,verbose=verbose)


    


class simple_self_att(layers.Layer):

  def __init__(self,**kwargs):
    super().__init__(**kwargs)
    self.conv = None
  def call(self, image):         
      if self.conv is None:
          self.conv = layers.Conv3D(image.shape[4],1,strides=(1,1,1))
          
      x = self.conv(image)
      g = (tf.sign(x)+1.0)*0.5 + 0.05*x
      
      return g*image
  
custom_layers['simple_self_att'] = simple_self_att


def createUnet_v3(depth=5,outK=1,feature_dim=None,nD=3,
                  padding='SAME',verbose=False,input_shape=None,self_att=False):
  if nD == 3:
      strides = [2,2,2]
      _convS = lambda *args, **kwargs: layers.Conv3D(*args, **kwargs)
      _convD = lambda *args, **kwargs: layers.Conv3D(*args, **kwargs,strides=strides)
      _convT = lambda *args, **kwargs: layers.Conv3DTranspose(*args, **kwargs,strides=strides)
  elif nD == 2:
      strides = [2,2]
      _convS = lambda *args, **kwargs: layers.Conv2D(*args, **kwargs)
      _convD = lambda *args, **kwargs: layers.Conv2D(*args, **kwargs,strides=strides)
      _convT = lambda *args, **kwargs: layers.Conv2DTranspose(*args, **kwargs,strides=strides)
    
  if self_att:
      nlin = simple_self_att
  else:
      nlin = layers.LeakyReLU
    
    
  def conv_down(fdim):
         return [ _convD(fdim,2,padding='SAME'), layers.BatchNormalization(), nlin()] 
  def conv_up(fdim):
         return [ _convT(fdim,2,padding='SAME'), layers.BatchNormalization(), nlin()] 
  def conv(outK):
         return [ _convS(fdim,3,padding='SAME'), layers.BatchNormalization(), nlin()] 
   
      
  if feature_dim is None:
      fdims = [16,16,32,32,64,64,128,128]

  if input_shape is not None:
     tmp = input_shape
     input_shape = []
     input_shape.extend(tmp)
      
      
  theLayers = {}
  for z in range(depth):
      
      
    fdim = fdims[z]
    id_d = str(1000 + z+1)
    id_u = str(2000 + depth-z+1)
    
    theLayers[id_d+"0_conv"] = conv(fdim)
    theLayers[id_d+"1_conv"] = [{'f': conv_down(fdim) } , {'f': conv(fdim), 'dest':id_u+"1_conv" }  ]
                
    
    theLayers[id_u+"0_conv"] = conv_up(fdim)
    theLayers[id_u+"1_conv"] = conv(fdim)
    
    if input_shape is not None:
        for r in range(len(input_shape)):
            input_shape[r] = input_shape[r]//strides[r]
            if input_shape[r] == 1:
                strides[r] = 1
                
  theLayers["9_final"] = _convS(outK,1,padding='SAME')
      
  return CNNblock(theLayers,verbose=verbose)




    


class binaryLookup(layers.Layer):

  def __init__(self,out=-1,**kwargs):
    super().__init__(**kwargs)
    self.encoding = None
    self.out = out

    
  def get_config(self):
        config = super().get_config().copy()
        config.update(
        {
            'out': self.out,
        } )    
        return config                  
    
  def call(self, image):         
      if self.encoding is None:
          self.N = image.shape[-1]
          if self.out == -1:
              self.out = self.N
          self.encoding = self.add_weight(shape=[2**self.N,self.out], 
                        initializer=tf.keras.initializers.RandomNormal, trainable=True,name=self.name)

      e = 2**tf.range(self.N)
      for k in range(len(image.shape)-1):
          e = tf.expand_dims(e,0)
      idx = tf.reduce_sum(tf.where(image>0,e,0),-1)
      out = tf.gather(self.encoding,idx)
      out = out * tf.reduce_mean(tf.abs(image),keepdims=True,axis=-1)
      
      return out

    
custom_layers['binaryLookup'] = binaryLookup


class simple_self_att(layers.Layer):

  def __init__(self,**kwargs):
    super().__init__(**kwargs)
    self.conv = None
  def call(self, image):         
      if self.conv is None:
          self.conv = layers.Conv3D(image.shape[4],1,strides=(1,1,1))
          
      x = self.conv(image)
      g = (tf.sign(x)+1.0)*0.5 + 0.05*x
      
      return g*image
  
custom_layers['simple_self_att'] = simple_self_att


def createUnet_v3(depth=5,outK=1,feature_dim=None,nD=3,
                  padding='SAME',verbose=False,input_shape=None,self_att=False):
  if nD == 3:
      strides = [2,2,2]
      _convS = lambda *args, **kwargs: layers.Conv3D(*args, **kwargs)
      _convD = lambda *args, **kwargs: layers.Conv3D(*args, **kwargs,strides=strides)
      _convT = lambda *args, **kwargs: layers.Conv3DTranspose(*args, **kwargs,strides=strides)
  elif nD == 2:
      strides = [2,2]
      _convS = lambda *args, **kwargs: layers.Conv2D(*args, **kwargs)
      _convD = lambda *args, **kwargs: layers.Conv2D(*args, **kwargs,strides=strides)
      _convT = lambda *args, **kwargs: layers.Conv2DTranspose(*args, **kwargs,strides=strides)
    
  if self_att:
      nlin = simple_self_att
  else:
      nlin = layers.LeakyReLU
    
    
  def conv_down(fdim):
         return [ _convD(fdim,2,padding='SAME'), layers.BatchNormalization(), nlin()] 
  def conv_up(fdim):
         return [ _convT(fdim,2,padding='SAME'), layers.BatchNormalization(), nlin()] 
  def conv(outK):
         return [ _convS(fdim,3,padding='SAME'), layers.BatchNormalization(), nlin()] 
   
      
  if feature_dim is None:
      fdims = [16,16,32,32,64,64,128,128]
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
    
    theLayers[id_d+"0_conv"] = conv(fdim)
    theLayers[id_d+"1_conv"] = [{'f': conv_down(fdim) } , {'f': conv(fdim), 'dest':id_u+"1_conv" }  ]
                
    
    theLayers[id_u+"0_conv"] = conv_up(fdim)
    theLayers[id_u+"1_conv"] = conv(fdim)
    
    if input_shape is not None:
        for r in range(len(input_shape)):
            input_shape[r] = input_shape[r]//strides[r]
            if input_shape[r] == 1:
                strides[r] = 1
                
  theLayers["9_final"] = _convS(outK,1,padding='SAME')
      
  return CNNblock(theLayers,verbose=verbose)




def createFFTnet(depth=5,outK=1,feature_dim=None,nD=3,kmax=2,
                  padding='SAME',verbose=False,input_shape=None):
  if nD == 3:
      strides = [2,2,2]
      _convD = lambda *args, **kwargs: layers.Conv3D(*args, **kwargs,strides=strides)
  elif nD == 2:
      strides = [2,2]
      _convD = lambda *args, **kwargs: layers.Conv2D(*args, **kwargs,strides=strides)
    
  nlin = layers.LeakyReLU

  def conv_down(fdim):
         return [ _convD(fdim,2,padding='SAME'), layers.BatchNormalization(), nlin()] 
         
  if feature_dim is None:
      fdims = [16,16,32,32,64,64,128,128]
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
    
    theLayers[id_d+"0_conv"] = conv_down(fdim)
    
                
  theLayers["9_final"] = iFFTlayer(out_shape=input_shape,out_fdim=outK,kmax=kmax)
      
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



    
class iFFTlayer(layers.Layer):
  def __init__(self,out_shape=[32,32,32],out_fdim=6,kmax=2,**kwargs):    
    super().__init__(**kwargs)
    self.nD = len(out_shape)
    self.out_shape = out_shape
    self.out_fdim = out_fdim
    self.kmax = kmax
    self.flatten = tf.keras.layers.Flatten()
    self.numP = (2*kmax+1)**self.nD *  out_fdim
    self.dense = tf.keras.layers.Dense(self.numP)
    

    
  def get_config(self):
        config = super().get_config().copy()
        config.update(
        {
            'out_shape': self.out_shape,
            'out_fdim': self.out_fdim,
            'kmax': self.kmax
        } )    
        return config                  
    
        
  def call(self, image):         
      x = self.flatten(image)
      #print(x.shape)
      
      x = self.dense(x)
      #print(x.shape)
      sh = [x.shape[0],self.out_fdim] + [(2*self.kmax+1)]*self.nD
      
      totsz = tf.math.reduce_prod(sh)
      
      xran = lambda x: tf.math.mod(tf.range(-self.kmax,self.kmax+1),x)
      
      out_shape2 = list(map(lambda x:x*2,self.out_shape))
      
      if self.nD == 3:
          I = tf.meshgrid(tf.range(sh[0]),tf.range(sh[1]),xran(out_shape2[0]),xran(out_shape2[1]),xran(out_shape2[2]),indexing='ij')
      else:
          I = tf.meshgrid(tf.range(sh[0]),tf.range(sh[1]),xran(out_shape2[0]),xran(out_shape2[1]),indexing='ij')
      I = list(map(lambda i:tf.reshape(i,[totsz,1]),I))
      I = tf.concat(I,1)
      
      
      X = tf.scatter_nd(I,tf.reshape(x,[x.shape[0]*x.shape[1]]),
                            [x.shape[0],self.out_fdim] + out_shape2)
      X = tf.complex(X,0.0)
      if self.nD == 3:
          X = tf.signal.fft3d(X)
          X = tf.math.real(X)+tf.math.imag(X)
          X = X[:,:,0:self.out_shape[0],0:self.out_shape[1],0:self.out_shape[2]]
          X = tf.transpose(X,[0,2,3,4,1])
      else:
          X = tf.signal.fft2d(X)
          X = tf.math.real(X)+tf.math.imag(X)
          X = X[:,:,0:self.out_shape[0],0:self.out_shape[1]]
          X = tf.transpose(X,[0,2,3,1])
            
      return X

custom_layers['iFFTlayer'] = iFFTlayer



    
class QMactivation(layers.Layer):
  def __init__(self, **kwargs):    
    super().__init__(**kwargs)
    
  def call(self, image):         
      x = image**2
      x = x / tf.reduce_sum(x,axis=-1,keepdims=True)
      return x
custom_layers['QMactivation'] = QMactivation


class QMembedding(layers.Layer):
  def __init__(self,numC,embedD,bias=1,**kwargs):    
    super().__init__(**kwargs)
    self.numC = numC
    self.embedD = embedD
    self.bias = bias
    self.isQMembedding = True
  
  def get_config(self):
    config = super().get_config().copy()
    config.update(
    {
        'numC': self.numC,
        'bias': self.bias,
        'embedD': self.embedD
    } )    
    return config
    
  def apply(self, r, full=False,bsize=16,params={}):
              
      ison = lambda y: (y in params and params[y])
      
      typ = tf.int64
      
      E = self.weight
      idx = tf.zeros(r.shape[0:-1],dtype=typ)
      maxp = tf.zeros(r.shape[0:-1],dtype=tf.float32)
      sump = tf.zeros(r.shape[0:-1],dtype=tf.float32)
      expval = tf.zeros(r.shape[0:-1],dtype=tf.float32)
      tmp = []
      if bsize > self.numC:
          bsize = self.numC
      nchunks = self.numC//bsize
      for k in range(nchunks):
          end = min(E.shape[0],(k+1)*bsize)
          if self.bias == 1:
              p = tf.einsum('...i,ji->...j',r,E[k*bsize:end,0:-1]) + E[k*bsize:end,-1]
          else:
              p = tf.einsum('...i,ji->...j',r,E[k*bsize:end,:]) 
                    
          p = tf.math.exp(p)
          sump = sump + tf.reduce_sum(p,-1)
          if ison('full'):
              tmp.append(tf.expand_dims(p,-1))
          if 'partial' in params:
              for i in params['partial']:
                  if i>=(k*bsize) and i < end:
                      tmp.append(tf.expand_dims(p[...,i-k*bsize],-1))
          if 'expval' in params:
              w = tf.cast(params['expval'][k*bsize:end],dtype=tf.float32)
              expval = expval + tf.einsum('...i,i->...',p,w)
                            
          Mp = tf.reduce_max(p,-1)  
          Mi = tf.argmax(p,-1)
            
          idx  = tf.where(Mp>maxp,Mi+k*bsize,idx)
          maxp = tf.where(Mp>maxp,Mp,maxp)

      maxp = tf.where(idx==0,0.0,maxp)  
      maxp = 10000.0*maxp/sump
      maxp = tf.expand_dims(maxp,-1)
      idx = tf.expand_dims(idx,-1)
  
      
      cast = lambda x: tf.cast(x,dtype=typ)
      retlist = [idx,cast(maxp)]
      
      if ison('embedding'):
          retlist.append(cast(50.0*r))
      if 'expval' in params:
          retlist.append(cast(tf.expand_dims(50.0*expval/sump,-1)))
      if ison('full') or 'partial' in params:
          tmp = 10000.0*tf.concat(tmp,-1)/tf.expand_dims(sump,-1)
          retlist.append(cast(tmp))
      return tf.concat(retlist,-1)
      

  def call(self, image):         
      if not hasattr(self,"weight"):
          self.weight = self.add_weight(shape=[self.numC,self.embedD+self.bias], 
                        initializer=tf.keras.initializers.RandomNormal, trainable=True,name=self.name)
      image.QMembedding = self
      return image

custom_layers['QMembedding'] = QMembedding

def QMloss(bias=1,num_samples=4,background_weight=1.0,typ='softmax'):
    
   def loss(x,y,class_weight=None, from_logits=True):

        E = y.QMembedding.weight

        if bias == 1:
            y = y[...,0:(E.shape[1]-1)]
            inp =  lambda e: tf.reduce_sum(e[...,0:e.shape[-1]-1]*y,axis=-1) + e[...,-1]
            inp2 = lambda e: tf.reduce_sum(e[...,0:e.shape[-1]-1]*tf.expand_dims(y,-2),axis=-1) + e[...,-1]
        else:
            y = y[...,0:(E.shape[1])]
            inp =  lambda e: tf.reduce_sum(e*y,axis=-1)
            inp2 = lambda e: tf.reduce_sum(e*tf.expand_dims(y,-2),axis=-1)

        nD = len(y.shape)-2        

        e = tf.squeeze(tf.gather(E,x))          
        e = inp(e)

        weight = tf.squeeze(tf.where(x==0,background_weight,1.0),-1)

        bsize = num_samples
        full = num_samples >= E.shape[0]
                
        if full:
            opp = tf.tile(tf.reshape(tf.range(E.shape[0]-1),[1]*(nD+1) + [E.shape[0]-1]),x.shape[0:-1]+[1])
        else:
            rshape = x.shape[0:-1] + [bsize]
            opp = tf.random.uniform(rshape,minval=0,maxval=E.shape[0]-1,dtype=tf.int32)
            
        opp = tf.where(opp>=x,opp+1,opp)
        opp = tf.squeeze(tf.gather(E,opp))
        opp = inp2(opp)
        if typ == 'softmax':
            maxl = tf.stop_gradient(tf.reduce_max(opp,axis=-1))
            maxl = tf.stop_gradient(tf.where(maxl>e,maxl,e))
            e = e - maxl
            opp = opp - tf.expand_dims(maxl,-1)
            sump = tf.math.exp(e) + tf.reduce_sum(tf.math.exp(opp),axis=-1)                
            return weight*(-e + tf.math.log(sump))
        elif typ == 'binary_hinge':
            threshold = 0.1
            l = tf.math.maximum(threshold-tf.expand_dims(e,-1),0)
            l = l + tf.math.maximum(threshold+opp,0)
            return weight*tf.reduce_sum(l,-1)
        elif typ == 'binary_hingemax':
            threshold = 0.2
            l1 = tf.math.maximum(threshold-e,0)
            l0 = tf.math.maximum(threshold+opp,0)
            return weight*(l1+tf.reduce_max(l0,-1))
        elif typ == 'hinge':
            threshold = 0.2
            l = tf.math.maximum(threshold-(tf.expand_dims(e,-1)-opp),0)
            return weight*tf.reduce_sum(l,-1)
        elif typ == 'hingemax':
            threshold = 0.2
            l = tf.math.maximum(threshold-(tf.expand_dims(e,-1)-opp),0)
            return weight*tf.reduce_max(l,-1)
            
        else:
            assert False, 'wrong loss typ'
            
            
   def loss___(x,y,class_weight=None, from_logits=True):
        nD = len(y.shape)-2        
        E = y.QMembedding.weight
        y = tf.math.atan(y)
        K = tf.einsum('ij,ik->jk',E,E)
        e = tf.squeeze(tf.gather(E,x))
        if False: # with bias
            y = y[...,0:e.shape[-1]-1]
            e = tf.reduce_sum(e[...,0:e.shape[-1]-1]*y,axis=-1) + e[...,-1]
            sz = list(y.shape)
            sz[-1] = 1
            ys = tf.zeros(sz)
            ys = tf.concat([y,ys],-1)
        else:
            e = tf.reduce_sum(e*y,axis=-1)
            ys = y
        if nD == 2:
            N = tf.einsum('bxyc,bxyd,cd->bxy',ys,ys,K)        
        else:
            N = tf.einsum('bxyzc,bxyzd,cd->bxyz',ys,ys,K)        
        
#        L = N - 2*e + 1
        
#        if class_weight is not None:
 #           L = L*tf.squeeze(tf.gather(class_weight,x))
  #      return 1-e/tf.math.sqrt(N)
 #       return L
     #   t = 0
     #   p = t+e**2
     #   return -tf.math.log(p) + tf.math.log(t*14+N)  # = -log(p/N)
        
        p = e**2
        return -tf.math.log(p) + tf.math.log(N) # = -log(p/N)
        
   return loss
    

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

  def __init__(self,nD=2,out=10,scfac=0.2,scaling=1.0,init='ct',typ='acos',normalize=False,trainable=True,dropout=False,ignoreInf=False,**kwargs):
    super().__init__(**kwargs)

    self.init = init
    self.scfac = scfac
    self.nD = nD
    self.normalize = normalize
    self.trainable = trainable
    self.dropout = dropout
    self.scaling = scaling
    self.ignoreInf = ignoreInf
    self.typ = typ
    
    self.dropout_layer = None

    if dropout is not None or dropout >0:
        self.dropout_layer = layers.Dropout(rate=dropout)
        self.normalize = False
    
    if init is None:
        if nD == 2:
            self.conv = layers.Conv2D(out,1,bias_initializer=tf.keras.initializers.GlorotUniform())
        else:
            self.conv = layers.Conv3D(out,1,bias_initializer=tf.keras.initializers.GlorotUniform())
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
            'typ':self.typ,
            'init': self.init,
            'out': self.out,
            'nD': self.nD,
            'dropout':self.dropout,
            'ignoreInf':self.ignoreInf
        } )    
        return config                  
      
  def call(self, image,training=False):    
      
      
      y = image*self.scaling
      ch = 0
      for k in range(y.shape[-1]):
        x = self.conv(y[...,k:k+1])
        if self.typ == 'acos':
            x = 1/tf.math.cosh(x)
        elif self.typ == 'atan':
            x = tf.math.atan(x)
        else:
            assert False, 'invalid typ'
        ch = ch + tf.where(tf.math.is_inf(y[...,k:k+1]),0.0,x)
      
      if self.dropout_layer is not None:
          ch = self.dropout_layer(ch,training=True)
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

class sigmoid_window(layers.Layer):

  def __init__(self,**kwargs):
      super().__init__(**kwargs)
  def call(self, image):       
      x = image[...,0:-1] * tf.sigmoid(image[...,-2:-1])
      return x

custom_layers['sigmoid_window'] = sigmoid_window
    



class warpLayer(layers.Layer):

  def __init__(self, shape, initializer=tf.keras.initializers.Constant(0), nD=0,typ=None,**kwargs):
    super().__init__(**kwargs)
    nD = len(shape)-1
    self.nD = nD
    self.typ=typ
    self.shape = shape
    self.shape_mult = tf.cast(shape[0:nD],dtype=tf.float32)
    for k in range(nD+1):
        self.shape_mult = tf.expand_dims(self.shape_mult,0)
    self.weight = self.add_weight(shape=shape, 
                        initializer=initializer, trainable=False,name=self.name)    
    
        
  def lin_interp(self,data,x):
    
          def frac(a):
              return a-tf.floor(a)
          if self.nD == 3:
              w = [frac(x[:,..., 0:1]),frac(x[:,..., 1:2]), frac(x[:,..., 2:3])]
          else:
              w = [frac(x[:,..., 0:1]),frac(x[:,..., 1:2]) ]
    
          def gather(d,idx,s):
              q = tf.convert_to_tensor(s)
              for k in range(self.nD+1):
                  q = tf.expand_dims(q,0)
              idx = idx + q
              weight = 1.0
              for k in range(self.nD):
                  if s[k] == 1:
                      weight = weight*w[k]
                  else:
                      weight = weight*(1-w[k])
                      
              return tf.gather_nd(d, idx ,batch_dims=0) * weight
          x = tf.cast(x,dtype=tf.int32)
          if self.nD == 3:
              res = gather(data,x,[0,0,0]) + gather(data,x,[1,0,0]) + gather(data,x,[0,1,0]) + gather(data,x,[0,0,1]) + gather(data,x,[0,1,1]) + gather(data,x,[1,0,1]) + gather(data,x,[1,1,0]) + gather(data,x,[1,1,1])
          else:
              res = gather(data,x,[0,0]) + gather(data,x,[1,0]) + gather(data,x,[0,1]) + gather(data,x,[1,1]) 
          return res
   

  def call(self, image):
     if self.typ == 'xyz':
        C = image
     else:
        C = np.pi - tf.math.atan2(image[...,1::2],-image[...,0::2])
        C = C/(np.pi*2)
     #C = tf.where(C>0.99,0.99,C)     
     C = C*tf.cast(self.shape_mult-1,dtype=tf.float32)
     C = tf.where(C<0,0.0,C)
     C = tf.where(C>self.shape_mult-2,self.shape_mult-2,C)
          
     nD = self.nD
     W = self.lin_interp(self.weight,C)
     W = W * 0.0005
    # m = tf.reduce_mean(W,keepdims=True,axis=range(1,nD+1))
     #sd = tf.math.reduce_std(W,keepdims=True,axis=range(1,nD+1))
    # W = W/(0.00001+m)
     
     
     return tf.concat([W,image],self.nD+1)
     
      
  def get_config(self):
        config = super().get_config().copy()
        config.update(
        {
            'nD': self.nD,
            'shape': self.shape,
            'typ':self.typ,
        } )    
        return config                  

custom_layers['warpLayer'] = warpLayer


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



def maxloss2d(y,x,from_logits=True):
    loss = tf.keras.losses.binary_crossentropy(tf.expand_dims(y,4),tf.expand_dims(x,4),from_logits=from_logits)
    return tf.reduce_max(tf.where(y<0.5,loss,0),axis=[1,2]) + tf.reduce_max(tf.where(y>0.5,loss,0),axis=[1,2])
        
def maxloss3d(y,x,from_logits=True):
    loss = tf.keras.losses.binary_crossentropy(tf.expand_dims(y,5),tf.expand_dims(x,5),from_logits=from_logits)
    return tf.reduce_max(tf.where(y<0.5,loss,0),axis=[1,2,3]) + tf.reduce_max(tf.where(y>0.5,loss,0),axis=[1,2,3])











def XXmaxloss3D(l,p):
    t=0.2
    x = tf.math.logical_and(l<0.5,p>-t)
    a = tf.reduce_max(tf.where(x,p+t,0),axis=[2,3,4]) 
    
    x = tf.math.logical_and(l>0.5,p<t)
    b = tf.reduce_max(tf.where(x,-p+t,0),axis=[2,3,4]) 
    return a+b
           

def Max_loss3D(losstype='bc',threshold=1,axis=[1,2,3]):
    
    def theloss(y,x,from_logits=True):
        if losstype=='bc':
            loss = tf.keras.losses.binary_crossentropy(tf.expand_dims(y,5),tf.expand_dims(x,5),from_logits=from_logits)
        elif losstype=='hinge':
            if not from_logits:
                assert False,"hinge only working for logits"
            loss = tf.keras.losses.hinge(tf.expand_dims(y,5),tf.expand_dims(x,5)/threshold)*threshold
        else:
            assert False,'losstype not available'
            
        a = tf.where(y<0.5,loss,0)
        a = tf.reduce_max(a,keepdims=True,axis=axis)

        b = tf.where(y>0.5,loss,0)
        b = tf.reduce_max(b,keepdims=True,axis=axis)
        
        return tf.reduce_mean(a+b,axis=[1,2,3])
    
    return theloss
        

def Maxpool_loss3D(K=1,losstype='bc',threshold=1):
    
    def theloss(y,x,from_logits=True):
        if losstype=='bc':
            loss = tf.keras.losses.binary_crossentropy(tf.expand_dims(y,5),tf.expand_dims(x,5),from_logits=from_logits)
        elif losstype=='hinge':
            if not from_logits:
                assert False,"hinge only working for logits"
            loss = tf.keras.losses.hinge(tf.expand_dims(y,5),tf.expand_dims(x,5)/threshold)*threshold
        else:
            assert False,'losstype not available'
            
        a = tf.where(y<0.5,loss,0)
        a = tf.nn.max_pool3d(a,ksize=(2*K+1),strides=K+1,padding='VALID')
        print(a.shape)
        a = tf.reduce_mean(a,axis=[1,2,3])
    
        b = tf.where(y>0.5,loss,0)
        b = tf.nn.max_pool3d(b,ksize=(2*K+1),strides=K+1,padding='VALID')
        b = tf.reduce_mean(b,axis=[1,2,3])
        
        return a+b
    
    return theloss

        
def topk_loss2(y,x,K=1,from_logits=True,losstype='bc',combi=False,mismatch_penalty=False,nD=3):
    sz = y.shape
    if nD==2:
        nvx = sz[1]*sz[2]
    else:
        nvx = sz[1]*sz[2]*sz[3]
    ncl = sz[-1]
    if losstype=='bc':
        if nD == 2:
            loss = tf.keras.losses.binary_crossentropy(tf.expand_dims(y,4),tf.expand_dims(x,4),from_logits=from_logits)
        else:
            loss = tf.keras.losses.binary_crossentropy(tf.expand_dims(y,5),tf.expand_dims(x,5),from_logits=from_logits)
    elif losstype=='hinge':
        if not from_logits:
            assert False,"hinge only working for logits"
        if nD == 2:
            loss = tf.keras.losses.hinge(tf.expand_dims(y,4),tf.expand_dims(x,4))
        else:
            loss = tf.keras.losses.hinge(tf.expand_dims(y,5),tf.expand_dims(x,5))
    

    fac = 0.01
    sumloss = 0.0
    if combi>0:
        sumloss = tf.reduce_mean(loss,axis=list(range(1,len(sz))))*ncl*combi/fac
        
    isz = [-1,nvx,ncl]
    loss = tf.reshape(loss,isz)
    x = tf.reshape(x,isz)
    y = tf.reshape(y,isz)
    
    vpos=y>0.5
    vneg=y<0.5

    pos = tf.where(vpos,loss,0)
    neg = tf.where(vneg,loss,0)
    
    for j in range(ncl):
        if K[0] == 'inf':
            valspos = pos[...,j]
        else:
            valspos,_ = tf.nn.top_k(pos[...,j],k=K[0])
            
        sumloss = sumloss + tf.reduce_sum(valspos,axis=1)/(1+tf.reduce_sum(tf.where(valspos>0,1.0,0.0),axis=1))

        if K[1] == 'inf':
            valsneg = neg[...,j]
        else:
            valsneg,_ = tf.nn.top_k(neg[...,j],k=K[1])
        sumloss = sumloss + tf.reduce_sum(valsneg,axis=1)/(1+tf.reduce_sum(tf.where(valsneg>0,1.0,0.0),axis=1))

    return tf.expand_dims(sumloss,1)*fac


        
def topk_loss(y,x,K=1,from_logits=True,losstype='bc',combi=False,mismatch_penalty=False,nD=3):
    sz = y.shape
    if nD==2:
        nvx = sz[1]*sz[2]
    else:
        nvx = sz[1]*sz[2]*sz[3]
    ncl = sz[-1]
    if losstype=='bc':
        if nD == 2:
            loss = tf.keras.losses.binary_crossentropy(tf.expand_dims(y,4),tf.expand_dims(x,4),from_logits=from_logits)
        else:
            loss = tf.keras.losses.binary_crossentropy(tf.expand_dims(y,5),tf.expand_dims(x,5),from_logits=from_logits)
    elif losstype=='hinge':
        if not from_logits:
            assert False,"hinge only working for logits"
        if nD == 2:
            loss = tf.keras.losses.hinge(tf.expand_dims(y,4),tf.expand_dims(x,4))
        else:
            loss = tf.keras.losses.hinge(tf.expand_dims(y,5),tf.expand_dims(x,5))

    rmean = 1

    fac = 0.01
    sumloss = 0.0
    if combi>0:
        sumloss = tf.reduce_mean(loss,axis=list(range(1,len(sz))))*ncl*combi/fac
        
    isz = [-1,nvx,ncl]
    loss = tf.reshape(loss,isz)
    x = tf.reshape(x,isz)
    y = tf.reshape(y,isz)
    
    vpos=y>0.5
    vneg=y<0.5

    if mismatch_penalty:    
        if from_logits:
           vpos = tf.math.logical_and(vpos,x<0)     # false negatives
           vneg = tf.math.logical_and(vneg,x>0)     # false positives
        else:
           vpos = tf.math.logical_and(vpos,x<0.5)
           vneg = tf.math.logical_and(vneg,x>0.5)
        
    pos = tf.where(vpos,loss,0)
    neg = tf.where(vneg,loss,0)
    
    for j in range(ncl):
        numpos = tf.cast(tf.reduce_sum(tf.where(vpos[...,j],1.0,0)),dtype=tf.int32)+1
        if K == "inf":
            sumloss = sumloss + tf.reduce_mean(pos[...,j],axis=1)            
        else:
            mK = tf.math.minimum(K,numpos)
            valspos,_ = tf.nn.top_k(pos[...,j],k=mK)
            if rmean:
                sumloss = sumloss + tf.reduce_sum(valspos,axis=1)/(1+tf.reduce_sum(tf.where(valspos>0,1.0,0.0),axis=1))
            else:                
                sumloss = sumloss + tf.reduce_mean(valspos,axis=1)

        numneg = tf.cast(tf.reduce_sum(tf.where(vneg[...,j],1.0,0)),dtype=tf.int32)+1
        if K == "inf":
            sumloss = sumloss + tf.reduce_mean(neg[...,j],axis=1)            
        else:
            mK = tf.math.minimum(K,numneg)
            valsneg,_ = tf.nn.top_k(neg[...,j],k=mK)
            if rmean:
                sumloss = sumloss + tf.reduce_sum(valsneg,axis=1)/(1+tf.reduce_sum(tf.where(valsneg>0,1.0,0.0),axis=1))
            else:                
                sumloss = sumloss + tf.reduce_mean(valsneg,axis=1)

    return tf.expand_dims(sumloss,1)*fac


def TopK_loss3D(K=1,losstype='bc',combi=False,mismatch_penalty=False,version=1):
    if version == 1:
        def loss(x,y,from_logits=True):
            return topk_loss(x,y,K=K,from_logits=from_logits,losstype=losstype,combi=combi,nD=3,mismatch_penalty=mismatch_penalty)
        return loss   
    else:
        def loss(x,y,from_logits=True):
            return topk_loss2(x,y,K=K,from_logits=from_logits,losstype=losstype,combi=combi,nD=3)
        return loss   
        

def TopK_loss2D(K=1,losstype='bc',combi=False,mismatch_penalty=False):
    def loss(x,y,from_logits=True):
        return topk_loss(x,y,K=K,from_logits=from_logits,losstype=losstype,combi=combi,nD=2,mismatch_penalty=mismatch_penalty)
    return loss   

    
    

#%%###############################################################################################

## A CNN wrapper to allow easy layer references and stacking

class CNNblock(layers.Layer):
  def __init__(self,theLayers=None,name=None,verbose=False,fromconfig=False,**kwargs):
    super().__init__(**kwargs)
    
    self.verbose = verbose
    if fromconfig:
        tmp = createCNNBlockFromObj(theLayers,custom_objects=custom_layers)
        self.theLayers = tmp.theLayers
    else:
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
            
          
  def get_config(self):
         config = super().get_config().copy()
         config.update(
         {
             'theLayers': dict(self.theLayers),
             'fromconfig': True
         } )    
         return config                  
     


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
            x=f(x,training=training)
            if self.verbose:
                print("   output_shape: " ,  x.shape)
            return x
        
    
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
            if self.verbose:
                print("---")
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

    if len(applyout) > 0 and not training:
        return tf.concat(applyout,nD+1)
    else:
        return x


custom_layers['CNNblock'] = CNNblock



def createCNNBlockFromObj(obj,custom_objects=None):

  def tolayer(x):
      if isinstance(x,dict):
          if 'keras_layer' in x:
              tmp = x['keras_layer']
              if 'groups' in tmp:
                  del tmp['groups']
              return layers.deserialize({'class_name':x['name'],'config':tmp},
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
      








