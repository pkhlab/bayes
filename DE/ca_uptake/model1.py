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
    ATP,op,caIn=y
    
    caEx=fixedParams[0]
    Kdatp=varDict[0]
    r=varDict[1]
    v=varDict[2]
    g=varDict[3]
    kleak=varDict[4]
    
    # ode
    dATPdt = -r*ATP
    dopdt = 1/(1+Kdatp/ATP)-v*op
    dcaIndt = caEx*g*op + kleak*(0.1-caIn)

    # the outputs (must be in the same order as ys above) 
    return [dATPdt,dopdt, dcaIndt]

# run simulation
def run_sim(varDict,fixedParams,y0s,ts):
    sol = solve_ivp(dydt,[0,ts[-1]],y0s,args=(varDict,fixedParams),t_eval=ts,method='LSODA').y[2]
    return sol

# objective function: sum of squared errors
def objective(varDict,fixedParams,y0s,ts,obs):
    sim = run_sim(varDict,fixedParams,y0s,ts)
    sim=minmax_scaling(sim)
    res = obs - sim
    return np.sum(res**2)

# read in observable data
def loadData(
            exptFile=None,
            simFile=None,
            frameRate=None,
            ):

    simData=pickle.load( open(simFile,'rb'))
    exptData=np.genfromtxt(exptFile, delimiter=',',skip_header=1)  
    nCells={'sim':len(simData['channel1'].processed),'expt':exptData.shape[1]}
    obs_data={'sim':[],'expt':[]}

    for key in obs_data.keys():
        n=nCells[key]
        for cell in range(n):
            if key == 'sim':
                caExpt=simData['channel1'].processed[cell]['Cai']
            else:
                caExpt=exptData[:,cell]
            obs_data[key].append(caExpt)
            
    return obs_data, nCells


# fitting function
def fitting(
            obs=None,
            ts=None,
            fixedParams=None,
            y0s=None,
            ):
    
    # bounds of the parameter search (essentially positive values)
    bounds=[(0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0)]
    # objective function
    obj = partial(objective, fixedParams=fixedParams, y0s=y0s, ts=ts, obs=obs)
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
            obs=obs_data[key][cell]
            obs=minmax_scaling(obs)
            stepSize=len(obs)
            ts=np.linspace(0,stepSize*frameRate[key],stepSize)
            
            if key == 'sim':
                y0s=[1.0, 0.1, obs[0]]
                fixedParams=[1.0e1]
            else:
                y0s=[1.0e-2, 0.1, obs[0]]
                fixedParams=[1.5]
            
            results_per_cell=[]
            for i in range(nIter):
                results=fitting(obs=obs,ts=ts,fixedParams=fixedParams,y0s=y0s)
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
    exptFile='./data/observed_data/ca_uptake/expt_data.csv',
    simFile='./data/observed_data/ca_uptake/sim_data.csv',
    frameRate={'sim':1.0,'expt':0.6},
    nParams=5,
    nIter=50,
    )    

