#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 11:37:27 2020

@author: reisertm
"""

from tensorflow.keras import layers
import tensorflow as tf


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
      

############################ biConv


class biConvolution(layers.Layer):

  def __init__(self, out_n=7, ksize=3, padding='SAME',transpose=False,nD=2,strides=None,**kwargs):
      
      
    super().__init__(**kwargs)
    self.out_n = out_n
    self.ksize = ksize
    self.padding = padding
    self.transpose = transpose
    self.nD = nD
    self.initializer = tf.random_normal_initializer(0, 0.05)
    self.num_alpha = 0
    self.isBi=True
    
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
              weight_shape = (self.num_alpha,self.ksize,self.ksize,self.M,self.N)                        
          else:
              weight_shape = (self.num_alpha,self.ksize,self.ksize,self.ksize,self.M,self.N)
             
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
        for k in range(self.num_alpha):
            kernel = self.weight[k,...]
            alpha = alphas[:,k:k+1]
            for j in range(self.nD):
                alpha = tf.expand_dims(alpha,j+2)
            c = conv(image, kernel, strides=self.strides, padding=self.padding)
            x = x+alpha*c
    
    return x


