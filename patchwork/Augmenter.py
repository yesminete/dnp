#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 11:37:27 2020

@author: reisertm
"""

import numpy as np
import torch
import torch.nn.functional as F

def Augmenter( morph_width = 150,
                morph_strength=0.25,
                flip = None ,
                scaling = None,
                normal_noise=0,
                repetitions=1,
                include_original=True):
                

    def augment(data,labels):

            if isinstance(data, np.ndarray):
                data = torch.from_numpy(data)
            if labels is not None and isinstance(labels, np.ndarray):
                labels = torch.from_numpy(labels)

            sz = data.shape
            if len(sz) != 4:
                raise ValueError('Augmenter supports only 2-D images with shape [batch, h, w, c]')
            nD = 2
    
            if not include_original:
                data_res = []
                if labels is not None:
                    labels_res = []
            else:
                data_res = [data]
                if labels is not None:
                    labels_res = [labels]
    
            for k in range(repetitions):
                X,Y = sampleDefField_2D(sz, data.device)
                data_ = interp2lin(data,Y,X)
                if labels is not None:
                    labels_ = interp2lin(labels,Y,X)
                
                if normal_noise > 0:
                    data_ = data_ + torch.randn_like(data_) * normal_noise
                
                if flip is not None:
                    for j in range(nD):
                        if flip[j]:
                            if np.random.uniform() > 0.5:
                                data_ = torch.flip(data_,dims=[j+1])
                                if labels is not None:
                                    labels_ = torch.flip(labels_,dims=[j+1])
                                
                
                
                data_res.append(data_)
                if labels is not None:            
                    labels_res.append(labels_)
            data_res = torch.cat(data_res,0)
            
            if labels is not None:            
                labels_res = torch.cat(labels_res,0)
                return data_res,labels_res
            else:            
                return data_res,None
            
    


    def gaussian_kernel(std, device):
        size = int(4 * std + 1)
        coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * std * std))
        g = g / g.sum()
        kernel = torch.outer(g, g)
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, size, size)

    def smooth_noise(shape, std, device):
        noise = torch.randn(1, 1, *shape, device=device)
        k = gaussian_kernel(std, device)
        return F.conv2d(noise, k, padding="same")[0, 0]

    def interp2lin(image, Y, X):
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)
        B, H, W, C = image.shape
        grid_x = 2 * X / (W - 1) - 1
        grid_y = 2 * Y / (H - 1) - 1
        grid = torch.stack((grid_x, grid_y), dim=-1)
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)
        out = F.grid_sample(image.permute(0, 3, 1, 2), grid, mode="bilinear", align_corners=True)
        return out.permute(0, 2, 3, 1)

    def sampleDefField_2D(sz, device):

        X, Y = torch.meshgrid(
            torch.arange(0, sz[2], dtype=torch.float32, device=device),
            torch.arange(0, sz[1], dtype=torch.float32, device=device),
            indexing="xy",
        )

        wid = morph_width / 4.0
        s = wid * wid * morph_strength
        dx = smooth_noise((sz[1], sz[2]), wid, device)
        dy = smooth_noise((sz[1], sz[2]), wid, device)

        scfacs = [1.0, 1.0]
        if scaling is not None:
            if isinstance(scaling, list):
                scfacs = [
                    1 + np.random.uniform(-1, 1) * scaling[0],
                    1 + np.random.uniform(-1, 1) * scaling[1],
                ]
            else:
                sciso = scaling * np.random.uniform(-1, 1)
                scfacs = [1 + sciso, 1 + sciso]

        cx = 0.5 * sz[2]
        cy = 0.5 * sz[1]
        nX = scfacs[0] * (X - cx) + cx + s * dx
        nY = scfacs[1] * (Y - cy) + cy + s * dy

        return nX, nY
    

    return augment


