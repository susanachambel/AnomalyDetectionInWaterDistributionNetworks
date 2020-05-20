# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 18:56:59 2020

@author: susan
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.matlib import repmat
from scipy import stats

from configuration import *
from data_selection import *

# Return sliding windows
def sliding_window(xx,k):
    # Function to generate boxes given dataset(xx) and box size (k)
    import numpy as np
    # generate indexes. O(1) way of doing it :)
    idx = np.arange(k)[None, :]+np.arange(len(xx)-k+1)[:, None]
    return xx[idx],idx


def compute_dpcca_others(cdata,k):
    # Input: cdata(nsamples,nvars), k: time scale for dpcca
    # Output: dcca, dpcca, corr, partialCorr
    #
    # Date(last modification): 02/15/2018
    # Author: Jaime Ide (jaime.ide@yale.edu)
    
    # Code distributed "as is", in the hope that it will be useful, but WITHOUT ANY WARRANTY;
    # without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
    # See the GNU General Public License for more details.
    

    # Define
    nsamples,nvars = cdata.shape

    # Cummulative sum after removing mean
    #cdata = signal.detrend(cdata,axis=0) # different from only removing the mean...
    cdata = cdata-cdata.mean(axis=0)
    xx = np.cumsum(cdata,axis=0)
    
    F2_dfa_x = np.zeros(nvars)
    allxdif = []
    # Get alldif and F2_dfa
    for ivar in range(nvars): # do for all vars
        xx_swin , idx = sliding_window(xx[:,ivar],k)
        nwin = xx_swin.shape[0]
        b1, b0 = np.polyfit(np.arange(k),xx_swin.T,deg=1) # linear fit (UPDATE if needed)
        
        #x_hat = [[b1[i]*j+b0[i] for j in range(k)] for i in range(nwin)] # Slower version
        x_hatx = repmat(b1,k,1).T*repmat(range(k),nwin,1) + repmat(b0,k,1).T
    
        # Store differences to the linear fit
        xdif = xx_swin-x_hatx
        allxdif.append(xdif)
        # Eq.4
        F2_dfa_x[ivar] = (xdif**2).mean()
    # Get the DCCA matrix
    dcca = np.zeros([nvars,nvars])
    for i in range(nvars): # do for all vars
        for j in range(nvars): # do for all vars
            # Eq.5 and 6
            F2_dcca = (allxdif[i]*allxdif[j]).mean()
            # Eq.1: DCCA
            dcca[i,j] = F2_dcca / np.sqrt(F2_dfa_x[i] * F2_dfa_x[j])   
    
    # Get DPCCA
    C = np.linalg.inv(dcca)
    
    # (Clear but slow version)
    dpcca = np.zeros([nvars,nvars])
    for i in range(nvars):
        for j in range(nvars):
            dpcca[i,j] = -C[i,j]/np.sqrt(C[i,i]*C[j,j])
    
    # DPCCA (oneliner version)
    #mydiag = np.sqrt(np.abs(np.diag(C)))
    #dpcca = (-C/repmat(mydiag,nvars,1).T)/repmat(mydiag,nvars,1)+2*np.eye(nvars)
    
    # Include correlation and partial corr just for comparison ;)
    # Compute Corr
    corr = np.corrcoef(cdata.T)
    # Get parCorr
    cov = np.cov(cdata.T)
    C0 = np.linalg.inv(cov)
    mydiag = np.sqrt(np.abs(np.diag(C0)))
    parCorr = (-C0/repmat(mydiag,nvars,1).T)/repmat(mydiag,nvars,1)+2*np.eye(nvars)
    
    return corr,parCorr,dcca,dpcca

     
# Simple time-series (from CrossValidated so that you can compare the results)
x1 = [-1.042061,-0.669056,-0.685977,-0.067925,0.808380,1.385235,1.455245,0.540762 ,0.139570,-1.038133,0.080121,-0.102159,-0.068675,0.515445,0.600459,0.655325,0.610604,0.482337,0.079108,-0.118951,-0.050178,0.007500,-0.200622]
x2 = [-2.368030,-2.607095,-1.277660,0.301499,1.346982,1.885968,1.765950,1.242890,-0.464786,0.186658,-0.036450,-0.396513,-0.157115,-0.012962,0.378752,-0.151658,0.774253,0.646541,0.311877,-0.694177,-0.412918,-0.338630,0.276635]
x3 = np.array(x1)+np.array(x2)**2 # ad hoc
x4 = np.array(x1)+np.array(x2)**3

# Plot
cdata = np.array([x1,x2,x3]).T
#plt.plot(cdata)
#plt.title('Sample time series')
#plt.legend(['$x_1$','$x_2$','$x_3$'])
#plt.show()


"""

# Define
nsamples,nvars = cdata.shape


# Cummulative sum after removing mean
cdata = cdata-cdata.mean(axis=0)
xx = np.cumsum(cdata,axis=0)
plt.plot(xx)
plt.title('Cummulative sum')
plt.legend(['$x_1$','$x_2$','$x_3$'])
print(xx.shape)
plt.show()
"""

# Example using the function
k = 6
corr,parCorr,dcca, dpcca = compute_dpcca_others(cdata,k)
print('Pearson:\n',corr)
print('PartialCorr:\n',parCorr)
print('DCCA(k={}):\n{}'.format(k,dcca))
print('DPCCA(k={}):\n{}'.format(k,dpcca))