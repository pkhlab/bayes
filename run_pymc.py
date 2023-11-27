import sys
import os
import yaml
sys.path.append('PATH')
import model
from scipy.interpolate import interp1d

 # Class for channels
class container():
    def __init__(self,fileName,
                    index=None,# none- single channel image; int otherwise
                    raw = None # array of data; usually leave undefined 
                    ):
        self.fileName=fileName
        self.index = index
        self.raw = raw

model=model.model()

def run(yamlFile=None):
    with open('PATH'+'/fittingParams.yaml') as yam:
        params = yaml.load(yam, Loader=yaml.FullLoader)
    
    file=params['file']
    customFile=params['customFile']
    nChannels=params['nChannels']
    y0sVals=params['y0sVals']
    mus=params['mus']
    sigmas=params['sigmas']
    frameRate=params['frameRate']
    normalize=params['normalize']
    smooth=params['smooth']
    smaWindows=params['smaWindows']
    nChains=params['nChains']
    init=params['init']
    sampler=params['sampler']
    advi=params['advi']
    sampleStep=params['sampleStep']   
    tuneStep=params['tuneStep']
    adviStep=params['adviStep']

    channels=[]
    
    # read in data
    for i in range(nChannels):
        # processed data from collaborators
        # for these data, they have a universal format as follows
        # single channel: cell 1, cell 2, cell 3
        # multiple channels: cell 1 channel 1, cell 1 channel 2, cell2 channel 1, cell2 channel 2
        if customFile is not None:
            exptData=np.genfromtxt(customFile, delimiter=',',skip_header=1)
            channel=exptData[:,cell*nChannels+i]

        # processed data from cell detection algo
        else:
            exptData=pickle.load( open(file,'rb'))
            key='channel{}'.format(i+1)
            channel=exptData[key].processed[cell]['Cai']

        channels.append(channel)
    
    # smoothing and normalization
    for i in range(nChannels):
        channel=channels[i]
        
        # smoothing
        if smooth:
            window=smaWindows[i]
            channel=sma.sma(window,channel)
            
            # interpolate other channels to make sure all channels have the same time step
            if i > 0:
                stepSize_curr=len(channels[i])
                ts_curr=np.linspace(0,stepSize_curr*frameRate,stepSize_curr)
                interp_func=interp1d(ts_curr,channel,fill_value='extrapolate')
                stepSize=len(channels[0])
                ts=np.linspace(0,stepSize*frameRate,stepSize)
                channel=interp_func(ts)

        # normalization
        if normalize:
            channel-=np.min(channel)
            channel/=np.max(channel)
    
    # run pymc
    model.pymc(y0sVals=y0sVals,yobs=channels,ts=ts,
               mus=mus,sigmas=sigmas,
               nChains=nChains,init=init,sampler=sampler,advi=advi,
               sampleStep=sampleStep,tuneStep=tuneStep,adviStep=adviStep,
               )


run()




