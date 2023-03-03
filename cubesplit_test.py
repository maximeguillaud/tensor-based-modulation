#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 05:51:45 2023

@author: Maxime Guillaud (Inria)

Basic benchmark of the modulation from the article "Cube-Split: A Structured 
Grassmannian Constellation for Non-Coherent SIMO Communications" 
by K.-H. Ngo, A. Decurninge, M. Guillaud, S. Yang,
IEEE Transactions on Wireless Communications, Vol. 19, No. 3, March 2020.

"""

import numpy as np
import matplotlib.pyplot as plt
import math
import cubesplit


# symbol length
T=8

# bit loading for real part (first row) and imaginary part (second row)
# each row should have T-1 values
Bi = 2*np.ones((2,T-1),dtype='int8')
#Bi = np.array([[4, 2, 2], [5, 3, 0]],dtype='int8')

cs=cubesplit.CubeSplitCodebook(Bi)

print(f'Codebook initialized. Each vector symbol has length {T} and encodes {cs.B} bits.')

MCiter=10000  # Monte Carlo iterations

cs_perf=[]

for SNR_dB in np.arange(00,30,2):

# Random binary data generation

    all_data = []
    all_codeword = []
    
    for i in range(MCiter):
    
        data = np.random.randint(2, size=cs.B)   # generate B random bits
        
        x = cs.mapper(data)    # Cube-Split mapping
        
        all_data.append(data)
        all_codeword.append(x)
    
    all_data = np.asarray(all_data)
    
 
# channel effects (multiplication with unit-norm scalar + additive noise)

    all_codeword = np.multiply(all_codeword,np.exp(np.random.uniform()*2j*math.pi)) # random unit-norm complex scalar
    noise = (np.random.normal(size=all_codeword.shape)+1j*np.random.normal(size=all_codeword.shape))/np.sqrt(2)*np.sqrt(1/T*np.power(10.,-SNR_dB/10))
    all_codeword+= noise
    
    
# Cube-Split demapping

    all_estimated_data = []
    
    for x in all_codeword:
        
        estimated_data = cs.approx_hard_demapper(x)    # approximate hard de-mapping
        all_estimated_data.append(estimated_data)
    
    
    bit_error_rate = np.count_nonzero(all_data-all_estimated_data) / np.prod(all_data.shape)
    sym_error_rate = np.count_nonzero(np.sum(np.abs(all_data-all_estimated_data),axis=1)) / all_data.shape[0]
    print("SNR = ",SNR_dB," dB, BER = ", bit_error_rate, ", SER = ", sym_error_rate)
    
    cs_perf.append([SNR_dB, sym_error_rate, bit_error_rate])
    
cs_perf=np.array(cs_perf)
plt.semilogy(cs_perf[:,0],cs_perf[:,1], label='SER')
plt.semilogy(cs_perf[:,0],cs_perf[:,2], label='BER')
plt.xlabel("SNR")
plt.ylabel("Error rate")
plt.legend()
