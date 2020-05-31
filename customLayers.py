#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 11:37:27 2020

@author: reisertm
"""

from tensorflow.keras import layers
import tensorflow as tf

import patchwork

custom_layers = {}




def createUnet_v1(depth=4,outK=1,multiplicity=1,feature_dim=5,nD=3,verbose=False):

  if nD == 3:
      _conv = layers.Conv3D
      _convT = lambda *args, **kwargs: layers.Conv3DTranspose(*args, **kwargs,strides=(2,2,2))
      _maxpool = layers.MaxPooling3D(pool_size=(2,2,2))
  elif nD == 2:
      _conv = layers.Conv2D
      _convT = lambda *args, **kwargs: layers.Conv2DTranspose(*args, **kwargs,strides=(2,2))
      _maxpool = layers.MaxPooling2D(pool_size=(2,2))
  
  def BNrelu():
      return [layers.BatchNormalization(), layers.LeakyReLU()]
  def conv_down(fdim):
       return _conv(fdim,3,padding='VALID') 
  def conv_up(fdim,even):
       return _convT(fdim,4+even,padding='VALID' )
  def conv(outK):
       return _conv(outK,4,padding='SAME') 
  
  if not isinstance(feature_dim,list):
      fdims=[]
      for z in range(depth):
          fdims.append(feature_dim*(1+z))     
  else:
      fdims = feature_dim

  theLayers = {}
  offs = [0,1,0,0]
  for z in range(depth):
    fdim = fdims[z]
    id_d = str(1000 + z+1)
    id_u = str(2000 + depth-z+1)
    theLayers[id_d+"conv0"] = [{'f': conv_down(fdim) } , {'f': conv(fdim), 'dest':id_u+"relu" }  ]
    for k in range(multiplicity-1):
        theLayers[id_d+"conv"+str(k+1)] = conv(fdim)            
    theLayers[id_d+"relu"] = BNrelu() + [_maxpool ]
    theLayers[id_u+"conv0"] = conv_up(fdim,offs[z])
    for k in range(multiplicity-1):
        theLayers[id_u+"conv"+str(k+1)] = conv(fdim)
    theLayers[id_u+"relu"] = BNrelu()
  theLayers["3000"] =  [layers.Dropout(rate=0.5), conv(outK)]
  return patchwork.CNNblock(theLayers,verbose=verbose)










class sigmoid_softmax(layers.Layer):

  def __init__(self,**kwargs):
    super().__init__(**kwargs)
  def call(self, image):         
      e = tf.math.exp(image)
      s = (1+tf.reduce_sum(e,axis=-1,keepdims=True))
      return  e/s
  
custom_layers['sigmoid_softmax'] = sigmoid_softmax


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
        for k in range(self.num_alpha):
            kernel = self.weight[k+offs,...]
            alpha = alphas[:,k:k+offs]
            for j in range(self.nD):
                alpha = tf.expand_dims(alpha,j+2)
            c = conv(image, kernel, strides=self.strides, padding=self.padding)
            x = x+alpha*c
    
    return x

custom_layers['biConvolution'] = biConvolution












