# Moving average function to smooth data
import numpy as np

def sma(window, # window size for smoothing
        data, # data to smooth
       ):
    weights=np.repeat(1, window)/window
    sma=np.convolve(data, weights, 'valid')
    
    return sma
