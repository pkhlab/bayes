import sys
import os
import yaml
import argparse
import importlib.util
import numpy as np
import sma

def data_preprocessing(
    data_file=None, # observed data file
    nChannels=1,
    smooth=False,          # whether to smooth data
    smoothWindows={'channel1':3,'channel2':5}, 
    frameRate=None,        # experimental frame rate
    normalize=False,
    ):
    
    # read in observed data
    raw_data=np.genfromtxt(data_file, delimiter=',')
    if nChannels>1:
        nCells=int(raw_data.shape[1]/nChannels)
    else:
        nCells=raw_data.shape[1]

    # package raw data into a dictionary
    obs_data=dict() # observed data
    y0s=dict()      # initial values

    for i in range(nChannels):
        key='channel{}'.format(i+1)
        ys=[]
        y0=[]

        for cell in range(nCells):
            if nChannels>1:
                channel_data=raw_data[:,cell]
            else:
                channel_data=raw_data[:,cell*nChannels+i]

            # smooth noisy data
            if smooth:
                window_size=smoothWindows[key]
                channel_data=sma.sma(window_size,channel_data)
                stepSize=len(channel_data)
                ts=np.linspace(0,stepSize*frameRate,stepSize)
                if i==0:
                    ts_pm=ts

                # to ensure all channels have the same time steps, we interpolate other channels over channel1 times
                if nChannels>1 and i>1:
                    ref_data=obs_data['channel1'][:,cell]
                    stepSize_ref=len(ref_data)
                    ts_ref=np.linspace(0,stepSize_ref*frameRate,stepSize_ref)
                    interp_func=interp1d(ts,channel_data,fill_value='extrapolate')
                    channel_data=interp_func(ts_ref)
            
            else:
                stepSize=len(channel_data)
                ts=np.linspace(0,stepSize*frameRate,stepSize)
                ts_pm=ts
            
            # normalization
            if normalize:
                channel_data-=np.min(channel_data)
                channel_data/=np.max(channel_data)

            ys.append(channel_data)
            y0.append(channel_data[0])

        ys=np.array(ys).T
        y0=np.array(y0).T
        obs_data[key]=ys
        y0s[key]=y0
        
    return obs_data, y0s, ts_pm


def run(yamlFile=None):
    
    # pass in command line argument inputs
    parser = argparse.ArgumentParser(description="define model and params")
    parser.add_argument('--model', type=str, help='path to model.py', required=True)
    parser.add_argument('--param', type=str, help='path to param yml file', required=True)
    parser.add_argument('--data', type=str, help='path to data file', required=True)
    args = parser.parse_args()
    
    # import model.py
    model_path=os.path.abspath(args.model)
    model_dir, model_file = os.path.split(model_path)
    model_name, model_ext = os.path.splitext(model_file)
    spec = importlib.util.spec_from_file_location(model_name, model_path)
    model_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model_module)
    model=model_module.model()
    
    # import observed data file
    data_file=os.path.abspath(args.data)
    
    # import param yaml file
    yml_file=os.path.abspath(args.param)
    with open(yml_file, 'r') as yam:
        params = yaml.load(yam, Loader=yaml.FullLoader)

    nChannels=params['nChannels']
    y0sVals=params['y0sVals']
    hyper_mu=params['hyper_mu']
    hyper_sigma=params['hyper_sigma']
    frameRate=params['frameRate']
    normalize=params['normalize']
    smooth=params['smooth']
    smoothWindows=params['smoothWindows']
    K=params['K']
    nChains=params['nChains']
    sampler=params['sampler']
    advi=params['advi']
    sampleStep=params['sampleStep']   
    tuneStep=params['tuneStep']
    adviStep=params['adviStep']
    
    # process raw observed data if needed (smoothing, normalization)
    obs_data,y0s,ts=data_preprocessing(data_file=data_file,nChannels=nChannels,
                                      smooth=smooth,smoothWindows=smoothWindows,
                                      normalize=normalize,frameRate=frameRate,)

    # run Bayesian inference with PyMC
    results=model.pymc(obs_data=obs_data,y0s=y0s,ts=ts,K=K,
                       y0sVals=y0sVals,hyper_mu=hyper_mu,hyper_sigma=hyper_sigma,normalize=normalize,
                       nChains=nChains,sampleStep=sampleStep,tuneStep=tuneStep,adviStep=adviStep,
                       sampler=sampler,advi=advi,
                       )


run()




