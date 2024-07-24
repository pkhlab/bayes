import numpy as np
import pickle
import pymc as pm
import arviz as az
import sunode
from sunode.wrappers.as_pytensor import solve_ivp
import os
import pandas as pd
import pytensor

    
class model():
    def __init__(self):
        #
        # All necessary parameters for the ode model defined below via yaml
        # 
        self.params={
            'Kdatp': 0.005,
            'kdecay': 1.0e0,
            'kclose': 6.0e-1,
            'kflux': 1.0e0,
            'kleak': 5.0e0,
            }       
    
    def minmax_scaling(self,ar):
        ar-=ar.min()
        ar/=ar.max()
        return ar
    
    # pymc only takes ode with params in a list rather than a dictionary
    def dydt(self,t,y,params):
        dcytodt = params.kRyr * y.ER2 - params.kSERCA * y.cyto
        dER2dt = params.kSERCA * y.cyto - params.kRyr * y.ER2
        
        return {'cyto': dcytodt, 'ER2': dER2dt}
    
    def pymc(
        self,
        obs_data=None,
        y0s=None,
        ts=None,
        y0sVals=None,
        K=3,                   # number of components in the mixture model
        hyper_mu=None,
        hyper_sigma=None,
        normalize=True,        # whether to normalize data
        nChains=5,             # number of chains to run pymc
        sampler=None,          # if none, use NUTS+jitter by pymc default
        advi=True,             # whether to use advi to initialize mcmc
        sampleStep=500,        
        tuneStep=500,
        adviStep=30000,
        ):  
        
        nCells=len(y0s['channel1'])
        nChannels=len(obs_data.keys())
        pymcModel=self.dydt

        # define pymc model
        with pm.Model() as pm_model:    

            # component weight prior
            w=pm.Dirichlet('w', a=np.ones(K))
            
            # hyperparam prior
            mu_kSERCA = pm.Lognormal('mu_kSERCA', mu=pm.math.log(hyper_mu['kSERCA']), sigma=hyper_sigma['kSERCA'], shape=K) 
            mu_kRyr = pm.Lognormal('mu_kRyr', mu=pm.math.log(hyper_mu['kRyr']), sigma=hyper_sigma['kRyr'], shape=K) 
            
            sigma_kSERCA = pm.HalfNormal('sigma_kSERCA', sigma=1e-1, shape=K)
            sigma_kRyr = pm.HalfNormal('sigma_kRyr', sigma=1e-1, shape=K)
            
            kSERCA_lognormal_dist= pm.Lognormal.dist(mu=pm.math.log(mu_kSERCA), sigma=sigma_kSERCA, shape=K)
            kRyr_lognormal_dist= pm.Lognormal.dist(mu=pm.math.log(mu_kRyr), sigma=sigma_kRyr, shape=K)
            
            # param prior
            kSERCA = pm.Mixture('kSERCA', w=w, comp_dists=kSERCA_lognormal_dist, shape=nCells)
            kRyr = pm.Mixture('kRyr', w=w, comp_dists=kRyr_lognormal_dist, shape=nCells)    
            
            # initial values of ODE states
            y0s_pm={'cyto': (y0s['channel1'], (nCells,)),
            'ER2': (y0s['channel2'], (nCells,))}
            
            # params
            params = {'kSERCA': (kSERCA, (nCells)),
                'kRyr': (kRyr, (nCells)),
                'useless': (np.array(1.0e0), ()) # bug of sunode: need this to make it work
                }
            
            # ODE simulation
            solution, _, problem, solver, _, _ = solve_ivp(
            y0=y0s_pm,
            params=params,
            rhs=pymcModel,
            tvals=ts,
            t0=ts[0],
            )          
            sim_data=dict()
            if nChannels>1:
                sim_data['channel1'] = pm.Deterministic('sim1', solution['cyto']) 
                sim_data['channel2'] = pm.Deterministic('sim2', solution['ER2'])
            else:
                sim_data['channel1'] = pm.Deterministic('sim1', solution['caIn'])  
            
            # normalization
            if normalize:
                for i in range(nChannels):
                    key='channel{}'.format(i+1)
                    sim=sim_data[key]
                    # pytensor.map does not specify dimension specification. So we need to do transpose back and forth
                    sim=sim.T
                    sim,_ = pytensor.map(fn=self.minmax_scaling, sequences=sim)
                    sim=sim.T
                    sim_data[key]=sim
                
            # compute likelihood
            sigma=pm.HalfCauchy('sigma',0.1) 
            Y1=pm.Normal("Y1", mu=sim_data['channel1'], sigma=sigma, observed=obs_data['channel1'],)
            if nChannels>1:
                Y2=pm.Normal("Y2", mu=sim_data['channel2'], sigma=sigma, observed=obs_data['channel2'],)
            
            # sampling
            if advi:
                trace=pm.sample(sampleStep,tune=tuneStep,target_accept=0.99,nuts_sampler="nutpie",
                                chains=nChains,init='advi',n_init=adviStep,step=sampler,compute_convergence_checks=False,
                               return_inferencedata=True,idata_kwargs = {'log_likelihood': True})
            else:
                trace=pm.sample(sampleStep,tune=tuneStep,target_accept=0.99,nuts_sampler="nutpie",
                                chains=nChains,step=sampler,compute_convergence_checks=False,
                               return_inferencedata=True,idata_kwargs = {'log_likelihood': True})
        
        # save results
        path=os.getcwd()
        trace.to_netcdf(path+'/trace.nc')        
        df=trace.to_dataframe()
        df.to_csv('results.csv')
        
        return trace 
                 

    