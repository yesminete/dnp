#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 15:29:09 2020

@author: reisertm
"""

import tensorflow as tf



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


