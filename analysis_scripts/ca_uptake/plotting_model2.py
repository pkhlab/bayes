import sys
import os
import matplotlib
# needed if running as standaline 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import yaml
import statsmodels.api as sm
import seaborn as sns
from scipy.interpolate import interp1d
import pickle
import numpy as np
sys.path.append('/home/xfang2/repos/celldetection')
import sma
from scipy.integrate import solve_ivp
import pandas as pd
from scipy.stats import lognorm

class container():
    def __init__(self,fileName,
                    index=None,# none- single channel image; int otherwise
                    raw = None # array of data; usually leave undefined 
                    ):
        self.fileName=fileName
        self.index = index
        self.raw = raw       

# dydt function
def dydt(t,y,params):
    ATP,caIn=y
    
    caEx=params['caEx']
    r=params['r']
    g=params['g']
    kleak=params['kleak']
    
    # ode
    dATPdt = -r*ATP
    dcaIndt = caEx*g*ATP + kleak*(0.1-caIn)

    # the outputs (must be in the same order as ys above) 
    return [dATPdt,dcaIndt]

# normalization function
def minmax_scaling(data):
    data-=np.min(data)
    data/=np.max(data)
    
    return data

def reorder(df,ref_param=None,paramList=None,K=3,nChains=5):
    # reordering based on reference param
    orders=[]
    reordered_vals=np.zeros((0,K))
    for chain in range(nChains):
        orders_chain=[]
        vals=[]
        for k in range(K):
            trace=df.loc[df['chain'] == chain, "('posterior', '{}[{}]', {})".format(ref_param,k,k)]
            vals.append(trace)
        vals=np.array(vals).T
        # reordering
        reordered=np.zeros_like(vals)
        for i in range(len(trace)):
            order = np.argsort(vals[i])
            reordered[i] = vals[i][order]
            orders_chain.append(order)
        orders.append(orders_chain)
        reordered_vals=np.append(reordered_vals,reordered,axis=0)
    for k in range(K):
        df['{}[{}]'.format(ref_param,k)]=reordered_vals[:,k]
    
    # apply reordering on other params
    for param in paramList:
        if param==ref_param:
            continue
        reordered_vals=np.zeros((0,K))
        for chain in range(nChains):
            vals=[]
            for k in range(K):
                trace=df.loc[df['chain'] == chain, "('posterior', '{}[{}]', {})".format(param,k,k)]
                vals.append(trace)
            vals=np.array(vals).T 
            # reordering
            reordered=np.zeros_like(vals)
            for i in range(len(trace)):
                order = orders[chain][i]
                reordered[i] = vals[i][order]
            reordered_vals=np.append(reordered_vals,reordered,axis=0)
        for k in range(K):
            df['{}[{}]'.format(param,k)]=reordered_vals[:,k]

            
def loadData(
            exptFile=None,
            simFile=None,
            frameRate={'sim':1.0, 'expt':0.6},
            nChains=5,
            K=3,
            model=1,
            infDataPath=None,
            reorderParam='mu_v',
            paramList=None,
            ):

    # observable data
    simData=pickle.load( open(simFile,'rb'))
    exptData=np.genfromtxt(exptFile, delimiter=',',skip_header=1)  
    data={'sim':[],'expt':[]}
    nCells={'sim':len(simData['channel1'].processed),'expt':exptData.shape[1]}
    for key in data.keys():
        n=nCells[key]
        for cell in range(n):
            if key == 'sim':
                caExpt=simData['channel1'].processed[cell]['Cai']
            else:
                caExpt=exptData[:,cell]
            data[key].append(caExpt)
            
    # inference data
    infData={}
    infData['sim']=[]
    infData['expt']=[]
    file=infDataPath+'/sim/hier/k{}/model{}/results.csv'.format(K,model)
    if os.path.isfile(file):
        df=pd.read_csv(file)
        reorder(df,ref_param=reorderParam,paramList=paramList,K=K,nChains=nChains)
        infData['sim']=df
    else:
        print('missing {}'.format(file))
    file=infDataPath+'/expt/hier/k{}/model{}/results.csv'.format(K,model)
    if os.path.isfile(file):
        df=pd.read_csv(file)
        reorder(df,ref_param=reorderParam,paramList=paramList,K=K,nChains=nChains)
        infData['expt']=df
    else:
        print('missing {}'.format(file))
        
    return data, infData, nCells

# Plot observable data
def plotObs(
            data=None,
            nCells=None,
            frameRate={'sim':1.0, 'expt':0.6},
            cmap=None,
            ):
    
    # simulated data
    rcParams['figure.dpi']=300
    rcParams['figure.figsize']=4,5
    rcParams['font.size']=17
    fig=plt.figure()
    key='sim'
    n=nCells[key]
    gradient=np.linspace(0,1,n)

    for cell in range(n):
        idx= int(cell/4)
        if cell%4==0:
            ax=fig.add_subplot(3,1,idx+1)
            ax.set_ylabel('[Ca]$_i$ (%)')

        if cell==n-1:
            ax.set_xlabel('time (s)')

        if cell<8:
            ax.set_xticklabels([])

        caExpt=data[key][cell]
        stepSize=len(caExpt)
        ts=np.linspace(0,stepSize*frameRate[key],stepSize)
        yobs=caExpt
        yobs-=np.min(yobs)
        yobs/=np.max(yobs)
        ax.plot(ts,yobs,color=cmap(gradient[cell]),alpha=0.8)

    plt.tight_layout()
    plt.gcf().savefig('data_simulated.png')
    
    rcParams['figure.figsize']=4,2
    rcParams['font.size']=17
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    ax.set_xlabel('time (s)')
    ax.set_ylabel('[Ca]$_i$ (%)')
    key='expt'
    n=nCells[key]
    gradient=np.linspace(0,1,n)

    for cell in range(n):
        caExpt=data[key][cell]
        stepSize=len(caExpt)
        ts=np.linspace(0,stepSize*frameRate[key],stepSize)
        yobs=caExpt
        yobs-=np.min(yobs)
        yobs/=np.max(yobs)
        ax.plot(ts,yobs,color=cmap(gradient[cell]),alpha=0.8)

    plt.tight_layout()
    plt.gcf().savefig('data_expt.png')

    
# Plot fits
def plotFits(
            data=None,
            nCells=None,
            infData=None,
            params=None,
            fixedParams=None,
            K=3,
            nChains=5,
            model=1,
            cmap=None,
            frameRate=None,
            ):

    rcParams['font.size']=17
    rcParams['figure.dpi']=300
    
    for key in data.keys():
        if key=='expt':
            rcParams['figure.figsize']=16,3.5
            nrows,ncols=2,4
        else:
            rcParams['figure.figsize']=16,5
            nrows,ncols=3,4
            
        n=nCells[key]
        df=infData[key]
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex='col', sharey='row')
        cell=0
        gradient=np.linspace(0,1,n)

        for row in range(nrows):
            for col in range(ncols):
                ax = axes[row, col]        

                if col == 0:  # Leftmost column
                    ax.set_ylabel('[Ca$^{2+}$]$_i$ (%)')
                if row == nrows - 1:  # Bottom row
                    ax.set_xlabel('time (s)')

                try:
                    channel1=data[key][cell]
                except:
                    ax.set_visible(False) 
                    continue

                channel1=minmax_scaling(channel1)
                stepSize=len(channel1)
                ts=np.linspace(0,stepSize*frameRate[key],stepSize)
                ax.plot(ts,channel1,lw=3,color='gray',alpha=0.5,linestyle='--')   

                channel1Min,channel1Max=np.min(channel1),np.max(channel1)
                y0=channel1[0]

                varDict,varDictLower,varDictUpper={},{},{}
                varDict['caEx'],varDictLower['caEx'],varDictUpper['caEx']=fixedParams[key]['caEx'],fixedParams[key]['caEx'],fixedParams[key]['caEx']
                for i,param in enumerate(params):
                    vals=np.array([]) # array to combine all chains
                    for chain in range(nChains):
                        trace=df.loc[df['chain'] == chain, "('posterior', '{}[{}]', {})".format(param,cell,cell)]
                        vals=np.append(vals,trace)
                    N=len(vals)
                    avg, lower, upper=np.mean(vals), np.sort(vals)[ int(0.05*N) ], np.sort(vals)[ int(0.95*N) ]
                    varDict[param],varDictLower[param],varDictUpper[param]=avg,lower,upper

                if key == 'sim':
                    y0s=[1.0, y0]
                else:
                    y0s=[1.0e-2, y0]
                ys = solve_ivp(dydt,[0,ts[-1]],y0s,args=(varDict,),t_eval=ts,method='LSODA').y
                caSim=ys[1] 
                ys = solve_ivp(dydt,[0,ts[-1]],y0s,args=(varDictLower,),t_eval=ts,method='LSODA').y
                caSimLower=ys[1] 
                ys = solve_ivp(dydt,[0,ts[-1]],y0s,args=(varDictUpper,),t_eval=ts,method='LSODA').y
                caSimUpper=ys[1] 
                # normalization
                caSim=minmax_scaling(caSim)
                caSimLower=minmax_scaling(caSimLower)
                caSimUpper=minmax_scaling(caSimUpper)
                # rescale
                caSim=caSim*(channel1Max-channel1Min)+channel1Min
                caSimLower=caSimLower*(channel1Max-channel1Min)+channel1Min
                caSimUpper=caSimUpper*(channel1Max-channel1Min)+channel1Min
                # area
                ax.fill_between(ts,caSimLower,caSimUpper,alpha=0.5,color=cmap(gradient[cell]),label='cell {}'.format(cell+1))

                cell+=1

                ax.legend(loc=1)
        plt.tight_layout()
        plt.savefig('model{}_fit_{}_k{}.png'.format(model,key,K))


# Plot posterior predictive checks
def plotPPC(
            data=None,
            nCells=None,
            infData=None,
            params=None,
            fixedParams=None,
            K=3,
            nChains=5,
            model=1,
            nSample=1000,
            cmap=None,
            frameRate=None,
            ):

    rcParams['font.size']=17
    rcParams['figure.dpi']=300
    
    for key in data.keys():
        if key=='expt':
            rcParams['figure.figsize']=16,3.5
            nrows,ncols=2,4
        else:
            rcParams['figure.figsize']=16,5
            nrows,ncols=3,4
            
        n=nCells[key]
        df=infData[key]
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex='col', sharey='row')
        cell=0
        gradient=np.linspace(0,1,n)

        for row in range(nrows):
            for col in range(ncols):
                ax = axes[row, col]        

                if col == 0:  # Leftmost column
                    ax.set_ylabel('[Ca$^{2+}$]$_i$ (%)')
                if row == nrows - 1:  # Bottom row
                    ax.set_xlabel('time (s)')

                try:
                    channel1=data[key][cell]
                except:
                    ax.set_visible(False) 
                    continue

                channel1=minmax_scaling(channel1)
                stepSize=len(channel1)
                ts=np.linspace(0,stepSize*frameRate[key],stepSize)  

                channel1Min,channel1Max=np.min(channel1),np.max(channel1)
                y0=channel1[0]
                
                
                for idx in range(nSample):
                    varDict={}
                    varDict['caEx']=fixedParams[key]['caEx']
                    for i,param in enumerate(params):
                        vals=np.array([]) # array to combine all chains
                        for chain in range(nChains):
                            trace=df.loc[df['chain'] == chain, "('posterior', '{}[{}]', {})".format(param,cell,cell)]
                            vals=np.append(vals,trace)
                        draw=np.random.choice(vals,replace=True)
                        varDict[param]=draw

                    if key == 'sim':
                        y0s=[1.0, y0]
                    else:
                        y0s=[1.0e-2, y0]
                    ys = solve_ivp(dydt,[0,ts[-1]],y0s,args=(varDict,),t_eval=ts,method='LSODA').y
                    caSim=ys[1] 
                    # normalization
                    caSim=minmax_scaling(caSim)
                    # rescale
                    caSim=caSim*(channel1Max-channel1Min)+channel1Min
                    # plot
                    ax.plot(ts,caSim,alpha=0.1,color=cmap(gradient[cell]),label='cell {}'.format(cell+1))
                    
                    if idx==0:
                        ax.legend(loc=1)
                
                ax.plot(ts,channel1,lw=3,color='gray',alpha=0.5,linestyle='--') 
                cell+=1

        plt.tight_layout()
        plt.savefig('model{}_ppc_{}_k{}.png'.format(model,key,K))
        
        
# Plot posterior
def plotPos(
            data=None,
            infData=None,
            nCells=None,
            params=None,
            hypers_mu=None,
            hypers_sigma=None,
            K=3,
            nChains=5,
            model=1,
            hyperLabels=None,
            labels=None,
            cmap=None,
            k_cols=None,
            ):
    
    rcParams['figure.dpi']=300
    rcParams['font.size']=15
    rcParams['figure.figsize']=15,3
    
    for key in data.keys():
        df=infData[key]
        n=nCells[key]
        gradient=np.linspace(0,1,n)

        # hypers
        fig=plt.figure()  
        for i,param in enumerate(params):    
            ax=fig.add_subplot(1,3,i+1)
            ax.set_title(hyperLabels[i],fontsize=25)
            ax.set_xlabel('value')

            prs=[]
            for iteration in range(1000):
                pr=np.random.lognormal(mean=np.log(hypers_mu[key]['mu_{}'.format(param)]),sigma=hypers_sigma[key]['mu_{}'.format(param)])
                prs.append(pr) 
            sns.kdeplot(prs,color='gray',fill=True).set(ylabel=None)

            for k in range(K):
                mu_vals=np.array([])
                for chain in range(nChains):
                    trace=df.loc[df['chain'] == chain, "mu_{}[{}]".format(param,k,k)]
                    mu_vals=np.append(mu_vals,trace)
                w_vals=np.array([])
                for chain in range(nChains):
                    trace=df.loc[df['chain'] == chain, "w[{}]".format(k)]
                    w_vals=np.append(w_vals,trace)
                sigma_vals=np.array([])
                for chain in range(nChains):
                    trace=df.loc[df['chain'] == chain, "sigma_{}[{}]".format(param,k,k)]
                    sigma_vals=np.append(sigma_vals,trace)
                w=np.mean(w_vals)
                mu=np.mean(mu_vals)
                sigma=np.mean(sigma_vals)

                ps=np.random.lognormal(mean=np.log(mu),sigma=sigma,size=1000)
                p,xs=np.histogram(ps,bins=500,density=True)

                density = lognorm.pdf(xs, scale=mu, s=sigma)
                plt.fill_between(xs, w * density, alpha=0.5, color=k_cols[k],label='component{}'.format(k+1))

                if i==0:
                    ax.set_ylabel('probability density')
        plt.legend(fontsize=10,loc=1)
        plt.tight_layout()
        plt.savefig('model{}_posterior_{}_hyper_k{}.png'.format(model,key,K))

        # params
        fig=plt.figure()  
        for i,param in enumerate(params):    
            ax=fig.add_subplot(1,3,i+1)
            ax.set_title(labels[i],fontsize=25)
            ax.set_xlabel('value')
            ax.set_ylabel('probability density')

            for cell in range(n):
                vals=np.array([])
                for chain in range(nChains):
                    trace=df.loc[df['chain'] == chain, "('posterior', '{}[{}]', {})".format(param,cell,cell)]
                    vals=np.append(vals,trace)
                sns.kdeplot(vals,color=cmap(gradient[cell]),fill=True).set(ylabel=None)

            if i==0:
                    ax.set_ylabel('probability density')

        plt.tight_layout()
        plt.savefig('model{}_posterior_{}_k{}.png'.format(model,key,K))


# Plot MCMC traces
def plotTrace(
             data=None,
             infData=None,
             nCells=None,
             K=3,
             nChains=5,
             model=1,
             params=None,
             hyperLabels=None,
             sigmaLabels=None,
             labels=None,
             cmap=None,
             k_cols=None,
            ):
    
    rcParams['figure.dpi']=300
    rcParams['font.size']=15
    
    for key in data.keys():
        df=infData[key]
        n=nCells[key]
        gradient=np.linspace(0,1,n)

        # hypers
        rcParams['figure.figsize']=15,5
        fig=plt.figure()

        ax=fig.add_subplot(1,4,1)
        ax.set_title('weight',fontsize=25)
        ax.set_xlabel('MCMC iter')
        ax.set_ylabel('value')

        for k in range(K):
            for chain in range(nChains):
                trace=df.loc[df['chain'] == chain, "w[{}]".format(k,k)]
                xs=range(1,len(trace)+1)
                ax.plot(xs,trace,color=k_cols[k],alpha=0.3)

        for i,param in enumerate(params):
            ax=fig.add_subplot(2,4,i+2)
            ax.set_title(hyperLabels[i],fontsize=25)
            ax.set_xlabel('MCMC iter')

            for k in range(K):
                for chain in range(nChains):
                    trace=df.loc[df['chain'] == chain, "mu_{}[{}]".format(param,k,k)]
                    xs=range(1,len(trace)+1)
                    ax.plot(xs,trace,color=k_cols[k],alpha=0.3)

        for i,param in enumerate(params):
            ax=fig.add_subplot(2,4,i+6)
            ax.set_title(sigmaLabels[i],fontsize=25)
            ax.set_xlabel('MCMC iter')
            for k in range(K):
                for chain in range(nChains):
                    trace=df.loc[df['chain'] == chain, "sigma_{}[{}]".format(param,k,k)]
                    xs=range(1,len(trace)+1)
                    ax.plot(xs,trace,color=k_cols[k],alpha=0.3)
        plt.tight_layout()
        plt.savefig('model{}_trace_{}_hyper_k{}.png'.format(model,key,K))

        # params
        rcParams['figure.figsize']=15,2.5
        fig=plt.figure()
        for i,param in enumerate(params):
            ax=fig.add_subplot(1,3,i+1)
            ax.set_title(labels[i],fontsize=25)
            ax.set_xlabel('MCMC iter')
            if i==0:
                ax.set_ylabel('value')

            for cell in range(n):
                for chain in range(nChains):
                    trace=df.loc[df['chain'] == chain, "('posterior', '{}[{}]', {})".format(param,cell,cell)]
                    xs=range(1,len(trace)+1)
                    ax.plot(xs,trace,color=cmap(gradient[cell]),alpha=0.3)
        plt.tight_layout()
        plt.savefig('model{}_trace_{}_k{}.png'.format(model,key,K))
    


# Plot autocorrelation
def plotCorr(
            data=None,
            infData=None,
            nCells=None,
            K=3,
            nChains=5,
            model=1,
            params=None,
            hyperLabels=None,
            sigmaLabels=None,
            labels=None,
            cmap=None,
            k_cols=None,
            ):
    
    rcParams['figure.dpi']=300
    rcParams['font.size']=15

    for key in data.keys(): 
        df=infData[key]
        n=nCells[key]
        gradient=np.linspace(0,1,n)

        rcParams['figure.figsize']=15,5
        fig=plt.figure()  
        ax=fig.add_subplot(1,4,1)
        ax.set_title('weight',fontsize=25)
        ax.set_xlabel('lag')
        ax.set_ylabel('autocorrelation')    
        for k in range(K):
            for chain in range(nChains):
                trace=df.loc[df['chain'] == chain, "w[{}]".format(k,k)]
                autocorr = sm.tsa.acf(trace,nlags=500,fft=False)
                ax.plot(autocorr,color=k_cols[k],alpha=0.5)    

        for i,param in enumerate(params):    
            ax=fig.add_subplot(2,4,i+2)
            ax.set_title(hyperLabels[i],fontsize=25)
            ax.set_xlabel('lag')   
            for k in range(K):
                for chain in range(nChains):
                    trace=df.loc[df['chain'] == chain, "mu_{}[{}]".format(param,k,k)]
                    autocorr = sm.tsa.acf(trace,nlags=500,fft=False)
                    ax.plot(autocorr,color=k_cols[k],alpha=0.5)     

        for i,param in enumerate(params):    
            ax=fig.add_subplot(2,4,i+6)
            ax.set_title(sigmaLabels[i],fontsize=25)
            ax.set_xlabel('lag')  
            for k in range(K):
                for chain in range(nChains):
                    trace=df.loc[df['chain'] == chain, "sigma_{}[{}]".format(param,k,k)]
                    autocorr = sm.tsa.acf(trace,nlags=500,fft=False)
                    ax.plot(autocorr,color=k_cols[k],alpha=0.5)     
        plt.tight_layout()
        plt.savefig('model{}_autocorr_{}_hyper_k{}.png'.format(model,key,K))

        rcParams['figure.figsize']=15,2.5
        fig=plt.figure()  
        for i,param in enumerate(params):    
            ax=fig.add_subplot(1,3,i+1)
            ax.set_title(labels[i],fontsize=25)
            ax.set_xlabel('lag')
            if i==0:
                ax.set_ylabel('autocorrelation')    

            for cell in range(n):
                for chain in range(nChains):
                    trace=df.loc[df['chain'] == chain, "('posterior', '{}[{}]', {})".format(param,cell,cell)]
                    autocorr = sm.tsa.acf(trace,nlags=500,fft=False)
                    ax.plot(autocorr,color=cmap(gradient[cell]),alpha=0.5)     
        plt.tight_layout()
        plt.savefig('model{}_autocorr_{}_k{}.png'.format(model,key,K))
    

# Plot joint posterior
def plotJoint(
            data=None,
            infData=None,
            nCells=None,
            params=None,
            hypers_mu=None,
            hypers_sigma=None,
            K=3,
            nChains=5,
            model=1,
            hyperLabels=None,
            labels=None,
            cmap=None,
            k_cols=None,
            ):
    
    rcParams['figure.dpi']=300
    rcParams['font.size']=15
    rcParams['figure.figsize']=15,15
    
    for key in data.keys():
        df=infData[key]
        n=nCells[key]
        gradient=np.linspace(0,1,n)
        nParams=len(params)

        draws=[]
        pos_density=[]
        pos_xs=[]
        for i,param in enumerate(params):
            draw=[] #np.array([])
            dens=[]
            xss=[]
            for k in range(K):
                mu_vals=np.array([])
                for chain in range(nChains):
                    trace=df.loc[df['chain'] == chain, "mu_{}[{}]".format(param,k,k)]
                    mu_vals=np.append(mu_vals,trace)
                w_vals=np.array([])
                for chain in range(nChains):
                    trace=df.loc[df['chain'] == chain, "w[{}]".format(k)]
                    w_vals=np.append(w_vals,trace)
                sigma_vals=np.array([])
                for chain in range(nChains):
                    trace=df.loc[df['chain'] == chain, "sigma_{}[{}]".format(param,k,k)]
                    sigma_vals=np.append(sigma_vals,trace)
                w=np.mean(w_vals)
                mu=np.mean(mu_vals)
                sigma=np.mean(sigma_vals)

                ps=np.random.lognormal(mean=np.log(mu),sigma=sigma,size=1000)
                p,xs=np.histogram(ps,bins=500,density=True)
                density = lognorm.pdf(xs, scale=mu, s=sigma)*w
                draw.append(ps)#=np.append(draw,ps)
                dens.append(density)
                xss.append(xs)
            draws.append(draw)
            pos_density.append(dens)
            pos_xs.append(xss)
        draws=np.array(draws)

        fig, axes = plt.subplots(nParams, nParams,)
        for i in range(nParams):
            for j in range(nParams):
                ax = axes[i, j]
                if i == j:
                    for k in range(K):
                        xs=pos_xs[j][k]
                        density=pos_density[j][k]
                        ax.fill_between(xs,density, alpha=0.5, color=k_cols[k],label='component{}'.format(k+1))
                    ax.set_xlabel(hyperLabels[i],fontsize=25)
                elif i > j:
                    for k in range(K):
                        ax.scatter(draws[j][k], draws[i][k], s=25, alpha=0.1, color=k_cols[k])
                    if j == 0:
                        ax.set_ylabel(hyperLabels[i],fontsize=25)
                    if i == nParams - 1:
                        ax.set_xlabel(hyperLabels[j],fontsize=25)
                else:
                    ax.set_visible(False)

        plt.tight_layout()
        plt.savefig('model{}_jointpos_{}_k{}.png'.format(model,key,K))
        
        
# Plot fit MSE
def plotMSE(
            data=None,
            infData=None,
            nCells=None,
            fixedParams=None,
            params=None,
            K=3,
            nChains=5,
            model=1,
            cmap=None,
            frameRate=None,
            ):
    
    mses={'sim':[],
          'expt':[]}
    nIter=1000

    for key in data.keys():
        n=nCells[key]
        df=infData[key]

        for cell in range(n):
            mse1=[]

            channel1=data[key][cell]
            channel1=minmax_scaling(channel1)
            stepSize=len(channel1)
            ts=np.linspace(0,stepSize*frameRate[key],stepSize)

            channel1Min,channel1Max=np.min(channel1),np.max(channel1)
            y0=channel1[0]

            varDict={}
            varDict['caEx']=fixedParams[key]['caEx']
            for j in range(nIter):
                for i,param in enumerate(params):
                    vals=np.array([]) # array to combine all chains
                    for chain in range(nChains):  
                        trace=df.loc[df['chain'] == chain, "('posterior', '{}[{}]', {})".format(param,cell,cell)]
                        vals=np.append(vals,trace)  
                    draw=np.random.choice(vals,replace=True)
                    varDict[param]=draw

                if key == 'sim':
                    y0s=[1.0, y0]
                else:
                    y0s=[1.0e-2, y0]
                ys = solve_ivp(dydt,[0,ts[-1]],y0s,args=(varDict,),t_eval=ts,method='LSODA').y
                sim1=ys[1] 

                # normalization
                sim1=minmax_scaling(sim1)

                # rescale
                sim1=sim1*(channel1Max-channel1Min)+channel1Min

                # MSE
                m1=np.mean((sim1-channel1)**2)
                mse1.append(m1)

            mses[key].append(mse1)
    
    
    rcParams['font.size']=8
    rcParams['figure.dpi']=300
    #bootstrapping
    nIter=500
    nSample=20

    for i,key1 in enumerate(mses.keys()):
        if key1=='sim':
            rcParams['figure.figsize']=4,2
        else:
            rcParams['figure.figsize']=3,2
        n=nCells[key1]
        gradient=np.linspace(0,1,n)

        fig=plt.figure()

        ax=fig.add_subplot(1,1,1)
        ax.set_xlabel('cell')
        ax.set_xticks(np.arange(n+1))
        ax.set_ylabel('MSE')
        if key1 == 'sim':
            ax.set_ylim([0.00001, 0.01])
            yloc=0.007
        else:
            ax.set_ylim([0.00002, 0.05])
            yloc=0.035

        vals=mses[key1]
        parts=ax.violinplot(vals,showextrema=False)
        for i,pc in enumerate(parts['bodies']):
            pc.set_facecolor(cmap(gradient[i]))
            pc.set_alpha(0.3)

        for cell in range(n):
            ys=vals[cell]
            xs=np.random.normal(cell+1, 0.04, len(ys))
            ax.scatter(xs,ys,s=1,color=cmap(gradient[cell]),alpha=0.1)

            # bootstrapping
            nData=len(ys)
            means=[]
            for k in range(nIter):
                draws=np.random.choice(ys,nSample)
                mean=np.mean(draws)
                means.append(mean)
            avg=np.mean(means)
            std=np.std(means)
            ax.plot(xs,np.repeat(avg,len(xs)),color='gray')
            ax.annotate(f"{avg:.2e}", xy=(cell+1, yloc),fontsize=6,
                horizontalalignment="center",rotation=90,
                       )

        plt.tight_layout()
        plt.gcf().savefig('model{}_mse_{}_k{}.png'.format(model,key1,K))


# run all plotting functions
def run(
        hypers_mu=None,
        hypers_sigma=None,
        fixedParams=None,
        hyperParams=None,
        params=None,
        paramList=None,
        hyperLabels=None,
        sigmaLabels=None,
        labels=None,
        truthVals=None,
        exptFile=None,
        simFile=None,
        frameRate={'sim':1.0, 'expt':0.6},
        nChains=5,
        K=3,
        model=1,
        infDataPath=None,
        reorderParam='mu_v',
        k_cols=None,
        cmap=None,
        ):

        
        cmap=matplotlib.colormaps[cmap]

        data,infData,nCells=loadData(exptFile=exptFile,simFile=simFile,frameRate=frameRate,
                                     nChains=nChains,K=K,model=model,infDataPath=infDataPath,
                                    reorderParam=reorderParam,paramList=paramList,)
    
    
        plotObs(data=data,nCells=nCells,frameRate=frameRate,cmap=cmap,)
    
    
        plotFits(data=data,nCells=nCells,infData=infData,params=params,frameRate=frameRate,
            fixedParams=fixedParams,K=K,nChains=nChains,cmap=cmap,model=model,)
        
        plotPPC(data=data,nCells=nCells,infData=infData,params=params,frameRate=frameRate,
            fixedParams=fixedParams,K=K,nChains=nChains,cmap=cmap,model=model,)    
    
        plotPos(data=data,nCells=nCells,infData=infData,params=params,
            hypers_mu=hypers_mu,hypers_sigma=hypers_sigma,K=K,nChains=nChains,hyperLabels=hyperLabels,labels=labels,
            cmap=cmap,k_cols=k_cols,model=model,)
        
        
        plotTrace(data=data,nCells=nCells,infData=infData,params=params,K=K,nChains=nChains,
             hyperLabels=hyperLabels,labels=labels,sigmaLabels=sigmaLabels,cmap=cmap,k_cols=k_cols,model=model,)
    
    
        plotCorr(data=data,nCells=nCells,infData=infData,params=params,K=K,nChains=nChains,
            hyperLabels=hyperLabels,labels=labels,sigmaLabels=sigmaLabels,cmap=cmap,k_cols=k_cols,model=model,)
            
        
        plotJoint(data=data,nCells=nCells,infData=infData,params=params,
            hypers_mu=hypers_mu,hypers_sigma=hypers_sigma,K=K,nChains=nChains,hyperLabels=hyperLabels,labels=labels,
            cmap=cmap,k_cols=k_cols,model=model,)
        
        
        plotMSE(data=data,nCells=nCells,infData=infData,params=params,K=K,nChains=nChains,
                fixedParams=fixedParams,cmap=cmap,model=model,frameRate=frameRate,)
        

Ks=[2,3,4]

for k in Ks:

    run(
        exptFile='/home/xfang2/backup/faust_backup/automagikFitting/pymc3/bv2/expt/KO53a_downsampled.csv',

        simFile='/home/xfang2/backup/faust_backup/automagikFitting/pymc3/bv2/sim/simData/trans_multiClass.pkl',

        infDataPath='/home/xfang2/backup/faust_backup/automagikFitting/pymc3/bv2',

        truthVals={
        'Kdatp': [0.6018578680105743, 0.418992673582289, 0.7075977665268708, 0.5156082914299116,0.52770890429079, 0.4778433721447173, 0.5712763514117395, 0.6108782802052417,0.5583819546620961, 0.4876909476488548, 0.553975613035248, 0.3217524639240968], 
        'r': [0.04279411266634989, 0.05887162940307739, 0.05859588413717417, 0.04363476495582651,0.06318151554180138, 0.04530694715294101, 0.056755540851223814, 0.031829727734098034,0.8921246739572529, 0.6974833466452512, 0.8267448783455709, 0.8531091625293673], 
        'v': [0.6520405283748758, 0.3584509559558973, 0.7995046442135764, 0.7789529829045482,0.6261958897767302, 0.7876659943842496, 0.5982807703340012, 0.6938668949698165,0.7482038177117576, 0.6779849155260277, 0.7593311433410809, 0.4620446899842402], 
        'g': [0.11906648256224861, 0.05957490359610059, 0.09331845268383807, 0.10004236729366973,0.12095157145785444, 0.12091876511255331, 0.11727434583369678, 0.09755816850304652,0.07196053436998312, 0.09798163600102172, 0.0890351510162629, 0.09710760983261232], 
        'kleak': [0.4864908070413172, 0.4760182746882032, 0.5689042372916447, 0.31077850020936837,0.051247129537682165, 0.046772051943917044, 0.05841674712996142, 0.07390960515463034,0.4818618298979314, 0.4468038277249674, 0.5009164475456336, 0.5891092924163002], 
        },

        fixedParams={'sim':{'caEx':1.0e1,'ATP':1.0},
                    'expt':{'caEx':1.5,'ATP':1.0e-2}},
        
        cmap='viridis',

        k_cols=['lightskyblue','deepskyblue','royalblue','tab:blue'],

        hypers_mu={
    'sim':{'mu_r': 4.0e-1, 'mu_g': 1.0e-1, 'mu_kleak': 5.5e-1},
    'expt':{'mu_r': 4.0e-2, 'mu_g': 5.0e-2, 'mu_kleak': 3.0e-1}},

        hypers_sigma={
    'sim':{'mu_r': 1.0e-1, 'mu_g': 1.0e-1, 'mu_kleak': 1.0e-1},
    'expt':{'mu_r': 1.0e-1, 'mu_g': 1.0e-1, 'mu_kleak': 1.0e-1}},

        hyperParams=['mu_r','mu_g','mu_kleak'],

        params=['r','g','kleak'],

        paramList=['w','mu_r','mu_g','mu_kleak','sigma_r','sigma_g','sigma_kleak'],

        hyperLabels=['$\mu_{k_{decay}}$', '$\mu_{k_{flux}}$', '$\mu_{k_{leak}}$'],
        
        sigmaLabels=['$\sigma_{k_{decay}}$', '$\sigma_{k_{flux}}$', '$\sigma_{k_{leak}}$'],
        
        labels=['k$_{decay}$', 'k$_{flux}$', 'k$_{leak}$'],

        frameRate={'sim':1.0, 'expt':0.6},

        nChains=5,

        K=k,

        model=2,

        reorderParam='mu_r',  
        
    )
        
        
        
    
