import numpy as np
import pickle
import pymc as pm
import arviz as az
import sunode
from sunode.wrappers.as_pytensor import solve_ivp
import os
import sys
sys.path.append('/home/xfang2/repos/celldetection')
import sma
import pandas as pd


"""
For now dydt for pymc are embedded under pymc(), which probably is not an ideal way to do it. It is now more a workaround
in order for theano op (i.e. interpFunc()) to work. Need to rewrite this part later
"""

    
class model():
    def __init__(self):
        #
        # All necessary parameters for the ode model defined below via yaml
        # 
        self.params={
            'Kdatp': 0.005,
            'r': 1.0e0,
            'v': 6.0e-1,
            'g': 1.0e0,
            'leakRate': 5.0e0,
            }       
    
    # pymc only takes ode with params in a list rather than a dictionary
    def dydt_pymc(self,t,y,params):
        dATPdt = - params.r * y.atp
        dopdt = 1 / (1 + params.Kdatp / y.atp) - params.v * y.op
        dcaIndt = params.caEx * params.g * y.op + params.leakRate * (0.1 - y.caIn)
        
        return {
                'atp': dATPdt,
                'op': dopdt,
                'caIn': dcaIndt}
    
    def pymc(
        self,
        y0sVals=None,
        yobs=None, 
        ts=None,
        mus=None,
        sigmas=None,
        nChains=5,             # number of chains to run pymc
        init=False,            # run initial simulation to get y0s for actual inference 
        sampler=None,          # if none, use NUTS+jitter by pymc default
        advi=False             # whether to use advi to initialize mcmc
        sampleStep=500,        
        tuneStep=500,
        adviStep=30000,
        ):  
        
        pymcModel=self.dydt_pymc
        obs=yobs[0] # 0 is the index of the channels
        caIn0=obs[0]

        with pm.Model() as pm_model:
            # define priors for parameters
            Kdatp = pm.Lognormal('Kdatp', mu=pm.math.log(mus['Kdatp']), sigma=sigmas['Kdatp'])
            r = pm.Lognormal('r', mu=pm.math.log(mus['r']), sigma=sigmas['r'])
            v = pm.Lognormal('v', mu=pm.math.log(mus['v']), sigma=sigmas['v'])
            g = pm.Lognormal('g', mu=pm.math.log(mus['g']), sigma=sigmas['g'])
            leakRate = pm.Lognormal('leakRate', mu=pm.math.log(mus['leakRate']), sigma=sigmas['leakRate'])       
            
            # initial states
            y0s={'atp': (np.array(y0sVals['ATP']), ()),
            'op': (np.array(y0sVals['op']), ()),
            'caIn': (np.array(caIn0), ())}
            caEx=mus['caEx']
            
            # params
            params = {'Kdatp': (Kdatp, ()),
                'r': (r, ()),
                'v': (v, ()),
                'g': (g, ()),
                'leakRate': (leakRate, ()),
                'caEx': (np.array(caEx), ()), # fixed param
                }
            
            # run simulations of proposed param sets
            solution, _, problem, solver, _, _ = solve_ivp(
            y0=y0s,
            params=params,
            rhs=pymcModel,
            tvals=ts,
            t0=ts[0],)                
            sim = pm.Deterministic('sim', solution['caIn'])     
            if normalize:
                sim-=sim.min()
                sim/=sim.max()
                
            # compute likelihood
            sigma=pm.HalfCauchy('sigma',0.1) 
            Y=pm.Normal("Y", mu=sim, sigma=sigma, observed=obs)
            
            # sampling
            if advi:
                trace=pm.sample(sampleStep,tune=tuneStep,target_accept=0.99,
                                chains=nChains,init='advi',n_init=adviStep,step=sampler,compute_convergence_checks=False,
                               return_inferencedata=True,idata_kwargs = {'log_likelihood': True})
            else:
                trace=pm.sample(sampleStep,tune=tuneStep,target_accept=0.99,chains=nChains,
                                step=sampler,compute_convergence_checks=False,
                               return_inferencedata=True,idata_kwargs = {'log_likelihood': True})
        
        # save results
        path=os.getcwd()
        trace.to_netcdf(path+'/trace.nc')        
        df=trace.to_dataframe()
        df.to_csv('results.csv')
        
        
                 

    
