#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 05:51:45 2023

@author: Maxime Guillaud (Inria)

Implementation of the vector modulation from the article "Cube-Split: A Structured 
Grassmannian Constellation for Non-Coherent SIMO Communications" 
by K.-H. Ngo, A. Decurninge, M. Guillaud, S. Yang,
IEEE Transactions on Wireless Communications, Vol. 19, No. 3, March 2020.

"""

import numpy as np
from statistics import NormalDist
import graycode

class CubeSplitCodebook:
    def __init__(self, Bi):
        self.Bi = Bi
        self.cumBi = np.cumsum(Bi)
        self.T = Bi.shape[1]+1
        
        # pre-compute some useful constants
        self.Bcell = np.ceil(np.log2(self.T)).astype(int)     # number of bits to encode the cell index
        self.B = self.Bcell + np.sum(self.Bi)       # total number of bits per codeword
        self.bitweights_real = [1<< np.arange(1,bi+1) for bi in self.Bi[0] ]    # weights from eq. (9)
        self.bitweights_imag = [1<< np.arange(1,bi+1) for bi in self.Bi[1] ]

        #print("Cube-split symbol length T = ", self.T, ", B = ", self.B, "bits/symbol")


    def mapper(self,data_bits):
        # split data bits according to the bit loading pattern
        split_bits = np.split(data_bits,self.cumBi)
        bits_real = split_bits[0:self.T-1]           # real coordinate bits
        bits_imag = split_bits[self.T-1:2*(self.T-1)]     # imag coordinate bits

        # Apply Gray bit-to-symbol mapping to the coordinate bits
        bits_real_gray = [ np.unpackbits(graycode.tc_to_gray_code(np.packbits(b,bitorder='little')), bitorder='little', count=self.Bi[0,i]) for i,b in enumerate(bits_real)]
        bits_imag_gray = [ np.unpackbits(graycode.tc_to_gray_code(np.packbits(b,bitorder='little')), bitorder='little', count=self.Bi[1,i]) for i,b in enumerate(bits_imag)]

        bits_cell = split_bits[2*(self.T-1)]         # cell bits
        
        # mapping of coordinate bits to (0,1) according to eq. (9)
        a_real =  np.divide( [1+np.dot(b,w) for b,w in zip(bits_real_gray,self.bitweights_real)],  1<<(self.Bi[0]+1) )   
        a_imag =  np.divide( [1+np.dot(b,w) for b,w in zip(bits_imag_gray,self.bitweights_imag)],  1<<(self.Bi[1]+1) )
    
        # transform constellation according to eqs. (12-13)
        w = [NormalDist().inv_cdf(ar)+1j*NormalDist().inv_cdf(ai) for ar,ai in zip(a_real,a_imag)]
        normw = np.abs(w)            # elementwise complex norm
        exp12 = np.exp(-normw**2/2)
        t=np.sqrt((1-exp12)/(1+exp12))*w/normw
     
        # map to the cell according to eq. (11)
        cell_idx = np.dot(bits_cell, 1 << np.arange(self.Bcell))
        t1=np.concatenate((t[0:cell_idx], [1], t[cell_idx:]))    
        x = t1/np.linalg.norm(t1)
        
        return x

    def approx_hard_demapper(self, x):
        est_cell_idx=np.argmax(abs(x))      # estimated cell index
        
        # eq. (15)
        est_t = x[[*range(est_cell_idx) , *range(est_cell_idx+1,self.T)]] / x[est_cell_idx]
    
        # eq. (14)
        normt = np.abs(est_t)
        est_w = np.sqrt(2*np.log((1+normt**2)/(1-normt**2)))*est_t/normt 
        
        est_a_real = [NormalDist().cdf(ew) for ew in np.real(est_w)]
        est_a_imag = [NormalDist().cdf(ew) for ew in np.imag(est_w)]
    
        # demapping: extract the integers corresponding to the coordinates within each cell
        int_A_real=np.rint((est_a_real*(1<<(self.Bi[0]+1))-1)/2)
        int_A_imag=np.rint((est_a_imag*(1<<(self.Bi[1]+1))-1)/2)

        # undo Gray mapping and recover the bits  
        est_bits_real = [np.flip(np.unpackbits(np.array([graycode.gray_code_to_tc(int(int_A_real[i]))], dtype='>i4').view(np.uint8)))[:self.Bi[0][i]] for i in range(self.T-1)]
        est_bits_imag = [np.flip(np.unpackbits(np.array([graycode.gray_code_to_tc(int(int_A_imag[i]))], dtype='>i4').view(np.uint8)))[:self.Bi[1][i]] for i in range(self.T-1)]
        est_bits_cell = np.flip(np.unpackbits(np.array([est_cell_idx], dtype='>i4').view(np.uint8)))[:self.Bcell]
        
        est_data = np.concatenate(est_bits_real + est_bits_imag + [est_bits_cell])
   
        return est_data

