#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
An implementation of Tensor-Based Modulation following the article:
    
A. Decurninge, I. Land and M. Guillaud, "Tensor-Based Modulation for Unsourced 
Massive Random Access," in IEEE Wireless Communications Letters, vol. 10, 
no. 3, pp. 552-556, March 2021, doi: 10.1109/LWC.2020.3037523.
    
Created on Sun Feb 26 23:40:57 2023

@author: Maxime Guillaud (Inria)
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import tensorly as tl

import cubesplit


#%% Parameters

TBM_shape = [32,32]     # tensor dimensions
bits_per_real_dim = 3   # bits per real dimension in each cube-split cell

T = np.prod(TBM_shape)  # block size (in symbols)
D = len(TBM_shape)      # number of tensor modes used by the TBM modulation - this is "d" in the article

Ka = 120                # number of active users
N = 50                  # number of Rx antennas
N0 = 0.001              # Channel AWGN variance

Ka_est = Ka             # number of users that the receiver is trying to decode (Ka_est = Ka if genie-aided)

#%% Initialize the parameters of the vector codebooks in each tensor mode

cs=[]    # list of vector codebooks
Bi = []  # number of bits per vector symbol

for m in range(D):    
    codebook = cubesplit.CubeSplitCodebook(bits_per_real_dim*np.ones((2,TBM_shape[m]-1),dtype='int64'))
    cs.append(codebook)
    print(f'Mode {m+1}: T[{m+1}] = {TBM_shape[m]}, cube-split codebook with B[{m+1}] = {codebook.B} bits/vector symbol')
    Bi.append(codebook.B)


#%% Data generation and TBM modulation

alldata = []    # binary data to transmit
true_TBM_factors = []    # TBM factors before the outer product

for m in range(D):    
    modedata = []   # payload bit streams 
    columns = []    # complex vectors after cube-split modulation
    for _ in range(Ka):
        data = np.random.randint(2, size=Bi[m])   # generate B random bits
        vector = (cs[m].mapper(data))
        vector/=np.linalg.norm(vector)
        columns.append(vector)
        modedata.append(data)

    true_TBM_factors.append(np.array(columns).transpose())
    alldata.append(modedata)
alldata = np.concatenate(alldata,axis=1)

#true_TBM_tensor = tl.cp_tensor.cp_to_tensor([np.ones(Ka), true_TBM_factors])     # used to compute Tx power
TBM_sequences = tl.tenalg.khatri_rao(true_TBM_factors)


#%% Channel Effects

true_channels = (np.random.normal(size=(N,Ka))+1j*np.random.normal(size=(N,Ka)))/np.sqrt(2)  # Complex normal Gaussian iid channel realizations

true_factors = true_TBM_factors + [true_channels]           
true_decomposition = [np.ones(Ka), true_factors]                # ideal tensor containing true modulated data and true channel, in its decomposed form
true_tensor = tl.cp_tensor.cp_to_tensor(true_decomposition)     # this directly computes the y_0 term in eq. (4)

noise = (np.random.normal(size=true_tensor.shape)+1j*np.random.normal(size=true_tensor.shape))/np.sqrt(2)

noisy_tensor = true_tensor + noise*np.sqrt(N0)                  # compute the received signal y as shown in eq. (4)


#%% TBM receiver

estimated_decomposition, errors = tl.decomposition.parafac(noisy_tensor, rank=Ka_est, normalize_factors=True, init='random',
                        return_errors=True, verbose=False, tol=1e-14, n_iter_max=200)       # compute the rank-Ka_est approximation, eq. (13)

#reconstructed_tensor = tl.cp_tensor.cp_to_tensor(estimated_decomposition)

est_alldata = []
for m in range(D):
    est_modedata = []
    for k in range(Ka_est):
        est_data = cs[m].approx_hard_demapper(estimated_decomposition.factors[m][:,k])       # hard demapping of the estimated vectors in each mode, eq. (15)
        est_modedata.append(est_data)
    est_alldata.append(est_modedata)
est_alldata = np.concatenate(est_alldata,axis=1)


#%% Attempt to recover the right user ordering for the purpose of error rate computation

# Estimate the alignments between detected and true signals 
alignments = np.empty((D+1,Ka,Ka_est))
for m in range(D+1):
    alignments[m,:,:] = np.abs(np.tensordot(true_factors[m],np.conj(estimated_decomposition.factors[m]),axes=(0,0)))

# Estimate the permutation between the users (σ in the article)
_, est_sigma = linear_sum_assignment(-np.sum(alignments,axis=0))


# Plot the pairwise alignments (scalar products between true and estimated tensor components) for all (true,estimated) pairs
# If the estimated components are accurate and signa was correctly estimated above, the result should be diagonally-dominant matrices in all modes
plt.figure(1)
plt.clf()
for m in range(D+1):
    plt.subplot(1,D+1,m+1)
    if m==D:
        plt.title('channels')
    else:
        plt.title(f'mode {m+1}')
    plt.imshow(alignments[m,:,est_sigma])
plt.suptitle('alignments of estimated vs. true tensor factors')


#%% Performance computation

Btot = np.sum(Bi)    # total bits per user
Eb = np.sum(np.abs(TBM_sequences**2))/Ka/Btot # energy per bit

bit_error_rate = np.count_nonzero(est_alldata[est_sigma,:]-alldata) / np.prod(alldata.shape)
block_errors_nb = np.count_nonzero(np.sum(np.abs(est_alldata[est_sigma,:]-alldata),axis=1))

print(f"Blocklength T = {T}, N = {N} Rx antennas, Ka = {Ka} users, {Btot} bits/user, aggregate spectral efficiency {np.sum(Bi)*Ka/T:.1f} bits/channel use, Eb/N0 = {10*np.log10(Eb/N0):.3f} dB")
print(f"BLER = {block_errors_nb/Ka:.6f} ({block_errors_nb}/{Ka}), BER = {bit_error_rate:.6f}")

