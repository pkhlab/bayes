import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
import pickle
import time
from functools import partial


# format data to be compatible w data processed by fit_multiplechannels.py
class container():
    def __init__(self,fileName,
                    index=None,# none- single channel image; int otherwise
                    raw = None # array of data; usually leave undefined 
                    ):
        self.fileName=fileName
        self.index = index
        self.raw = raw       

# normalization function
def minmax_scaling(data):
    data-=np.min(data)
    data/=np.max(data)
    
    return data

# dydt function
def dydt(t,y,varDict,fixedParams):
    cyto, ER1, ER2=y
    
    kSC=varDict[0]
    kRyr=varDict[1]
    kcomp=varDict[2]
    
    dcytodt = kRyr*ER2 - kSC*cyto
    dER1dt = kSC*cyto - kcomp*ER1
    dER2dt = kcomp*ER1 - kRyr*ER2
    
    return [dcytodt,dER1dt,dER2dt]

# run simulation
def run_sim(varDict,fixedParams,y0s,ts):
    sol1 = solve_ivp(dydt,[0,ts[-1]],y0s,args=(varDict,fixedParams),t_eval=ts,method='LSODA').y[0]
    sol2 = solve_ivp(dydt,[0,ts[-1]],y0s,args=(varDict,fixedParams),t_eval=ts,method='LSODA').y[2]
    return sol1, sol2

# objective function: sum of squared errors
def objective(varDict,fixedParams,y0s,ts,obs1, obs2):
    sim1, sim2 = run_sim(varDict, fixedParams, y0s, ts)
    sim1=minmax_scaling(sim1)
    sim2=minmax_scaling(sim2)
    res1 = obs1 - sim1
    res2 = obs2 - sim2
    return np.sum(res1**2)+np.sum(res2**2)

# read in observable data
def loadData(
            exptFile=None,
            simFile=None,
            frameRate=None,
            ):
    
    simData=pickle.load( open(simFile,'rb'))
    exptData=np.genfromtxt(exptFile, delimiter=',',skip_header=1)  
    nCells={'sim':len(simData['channel1'].processed),'expt':int(exptData.shape[1]/2)}
    obs_data={'sim':{'cyto':[],'ER':[]},'expt':{'cyto':[],'ER':[]}}

    for key in obs_data.keys():
        n=nCells[key]
        for cell in range(n):
            if key == 'sim':
                channel1=simData['channel1'].processed[cell]
                channel2=simData['channel2'].processed[cell]
            else:
                channel1=exptData[:,cell*2]
                channel2=exptData[:,cell*2+1]
                channel1=channel1[~np.isnan(channel1)]
                channel2=channel2[~np.isnan(channel2)]
            obs_data[key]['cyto'].append(channel1)
            obs_data[key]['ER'].append(channel2)
            
    return obs_data, nCells


# fitting function
def fitting(
            obs1=None,
            obs2=None,
            ts=None,
            fixedParams=None,
            y0s=None,
            ):
    
    # bounds of the parameter search (essentially positive values)
    bounds=[(0, 5.0), (0, 1.0e-1), (0, 5.0),]
    # objective function
    obj = partial(objective, fixedParams=fixedParams, y0s=y0s, ts=ts, obs1=obs1, obs2=obs2)
    # run fitting
    results = differential_evolution(obj,bounds).x
    
    return results

# run fitting
def run(
        exptFile=None,
        simFile=None,
        frameRate=None,
        nIter=50,
        nParams=5,
        ):
    
    # start timer
    startTime = time.time()

    # get observables
    obs_data,nCells=loadData(exptFile=exptFile,
                      simFile=simFile,
                      frameRate=frameRate,)
    
    keys=obs_data.keys() #'sim' and 'expt'
    
    # run fitting
    for key in keys:

        n=nCells[key]

        fit_results=[]
        for cell in range(n):
            obs1=obs_data[key]['cyto'][cell]
            obs2=obs_data[key]['ER'][cell]
            obs1=minmax_scaling(obs1)
            obs2=minmax_scaling(obs2)
            stepSize=len(obs1)
            ts=np.linspace(0,stepSize*frameRate[key],stepSize)
            
            if key == 'sim':
                y0s=[obs1[0], 10., obs2[0]]
            else:
                y0s=[obs1[0], 0.1, obs2[0]]
            
            fixedParams=None
            
            results_per_cell=[]
            for i in range(nIter):
                results=fitting(obs1=obs1,obs2=obs2,ts=ts,fixedParams=fixedParams,y0s=y0s)
                results_per_cell.append(results)
            results_per_cell=np.array(results_per_cell)
            fit_results.append(results_per_cell)
        fit_results=np.array(fit_results)
        
        for p in range(nParams):
            output_data=fit_results.T[p]
            np.savetxt("fit_results_{}_{}.csv".format(key,p), output_data, delimiter = ",")
    
 
    # end timer
    endTime = time.time()
    runTime = endTime - startTime
    print(f"Total run time: {runTime:.2f} seconds")
    
    
run(
    exptFile='./data/observed_data/cyto_ER_shuttling/expt_data.csv',
    simFile='./data/observed_data/cyto_ER_shuttling/sim_data.csv',
    frameRate={'sim':0.05,'expt':3.0e-2},
    nParams=3,
    nIter=50,
    )    

