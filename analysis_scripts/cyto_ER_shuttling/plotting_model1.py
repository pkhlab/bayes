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
    cyto, ER1, ER2=y
    
    kSC=params['kSC']
    kRyr=params['kRyr']
    kcomp=params['kcomp']
    
    dcytodt = kRyr*ER2 - kSC*cyto
    dER1dt = kSC*cyto - kcomp*ER1
    dER2dt = kcomp*ER1 - kRyr*ER2
    
    return [dcytodt,dER1dt,dER2dt]

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
    data={'sim':{'cyto':[],'ER':[]},'expt':{'cyto':[],'ER':[]}}
    nCells={'sim':len(simData['channel1'].processed),'expt':int(exptData.shape[1]/2)}
    for key in data.keys():
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
            data[key]['cyto'].append(channel1)
            data[key]['ER'].append(channel2)
            
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
            cmap1=None,
            cmap2=None,
            ):
    
    rcParams['figure.dpi']=300
    
    rcParams['figure.figsize']=8,5
    rcParams['font.size']=17
    fig=plt.figure()
    key='sim'
    n=nCells[key]
    gradient=np.linspace(0,1,n)

    for cell in range(n):
        idx= int(cell/4)
        if cell%4==0:
            ax=fig.add_subplot(3,2,idx*2+1)
            axr=fig.add_subplot(3,2,idx*2+2)
            ax.tick_params(axis='y', colors='forestgreen')
            axr.tick_params(axis='y', colors='firebrick')
            ax.set_ylabel('[Ca]$_i$ (%)',color='forestgreen')
            axr.set_ylabel('[Ca]$_{ER}$ (%)',color='firebrick')


        if cell<8:
            ax.set_xticklabels([])
            axr.set_xticklabels([])
        else:
            ax.set_xlabel('time (s)')
            axr.set_xlabel('time (s)')

        channel1=data[key]['cyto'][cell]
        channel2=data[key]['ER'][cell]
        stepSize=len(channel1)
        ts=np.linspace(0,stepSize*frameRate[key],stepSize)
        channel1-=np.min(channel1)
        channel1/=np.max(channel1)
        channel2-=np.min(channel2)
        channel2/=np.max(channel2)
        ax.plot(ts,channel1,color=cmap1(gradient[cell]),alpha=0.8)
        axr.plot(ts,channel2,color=cmap2(gradient[cell]),alpha=0.8)
    plt.tight_layout()
    plt.gcf().savefig('data_simulated.png')
    
    # experimental data
    rcParams['figure.figsize']=8,3
    rcParams['font.size']=17
    fig=plt.figure()
    ax=fig.add_subplot(121)
    axr=fig.add_subplot(122)
    ax.set_xticklabels([])
    ax.set_ylabel('[Ca]$_i$ (%)',color='forestgreen')
    axr.set_ylabel('[Ca]$_{ER}$ (%)',color='firebrick')
    ax.tick_params(axis='y', colors='forestgreen')
    axr.tick_params(axis='y', colors='firebrick')
    ax.set_xlabel('time (s)')
    axr.set_xlabel('time (s)')

    key='expt'
    n=nCells[key]
    gradient=np.linspace(0,1,n)

    for cell in range(n):
        channel1=data[key]['cyto'][cell]
        channel2=data[key]['ER'][cell]
        stepSize=len(channel1)
        ts=np.linspace(0,stepSize*frameRate[key],stepSize)
        channel1-=np.min(channel1)
        channel1/=np.max(channel1)
        channel2-=np.min(channel2)
        channel2/=np.max(channel2)
        ax.plot(ts,channel1,color=cmap1(gradient[cell]),alpha=0.8)
        axr.plot(ts,channel2,color=cmap2(gradient[cell]),alpha=0.8)
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
            cmap1=None,
            cmap2=None,
            frameRate=None,
            ):

    rcParams['font.size']=17
    rcParams['figure.dpi']=300

    for key in data.keys():
        if key=='expt':
            rcParams['figure.figsize']=16,6
            nrows,ncols=2,4
        else:
            rcParams['figure.figsize']=16,5
            nrows,ncols=3,4

        n=nCells[key]
        df=infData[key]
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols)#, sharex='col', sharey='row')
        cell=0
        gradient=np.linspace(0,1,n)

        for row in range(nrows):
            for col in range(ncols):
                ax = axes[row, col]    
                axr=ax.twinx()
                ax.tick_params(axis='y', colors='forestgreen')
                axr.tick_params(axis='y', colors='firebrick')

                if col == 0:  # Leftmost column
                    ax.set_ylabel('[Ca$^{2+}$]$_i$ (%)',color='forestgreen')    
                else:
                    ax.set_yticklabels([])
                if col == ncols-1:
                    axr.set_ylabel('[Ca$^{2+}$]$_{ER}$ (%)',color='firebrick')
                else:
                    axr.set_yticklabels([])
                if row == nrows - 1:  # Bottom row
                    ax.set_xlabel('time (s)')

                try:
                    channel1=data[key]['cyto'][cell]
                    channel2=data[key]['ER'][cell]
                except:
                    ax.set_visible(False) 
                    axr.set_visible(False) 
                    continue

                channel1=minmax_scaling(channel1)
                channel2=minmax_scaling(channel2)

                # raw data
                stepSize=len(channel1)
                ts=np.linspace(0,stepSize*frameRate[key],stepSize) 
                ax.plot(ts,channel1,lw=2,color='forestgreen',alpha=0.5,linestyle='--')  
                axr.plot(ts,channel2,lw=2,color='firebrick',alpha=0.5,linestyle='--')  

                # run sma on experimental data
                if key == 'expt':
                    # sma
                    window1=window2=5
                    channel1=sma.sma(window1,channel1)
                    channel2=sma.sma(window2,channel2)
                    stepSize1=len(channel1)
                    ts=np.linspace(0,stepSize1*frameRate[key],stepSize1)
                    stepSize2=len(channel2)
                    ts2=np.linspace(0,stepSize2*frameRate[key],stepSize2)
                    # to ensure the two channels have the same time steps, we interpolate channel2 over the time steps of channel1
                    interp_func=interp1d(ts2,channel2,fill_value='extrapolate')
                    channel2=interp_func(ts)              

                y01=channel1[0]
                y02=channel2[0]

                channel1Min,channel1Max,channel2Min,channel2Max=np.min(channel1),np.max(channel1),np.min(channel2),np.max(channel2)

                varDict,varDictLower,varDictUpper={},{},{}
                for i,param in enumerate(params):
                    vals=np.array([]) # array to combine all chains
                    for chain in range(nChains):  
                        trace=df.loc[df['chain'] == chain, "('posterior', '{}[{}]', {})".format(param,cell,cell)]
                        vals=np.append(vals,trace)
                    N=len(vals)
                    avg, lower, upper=np.mean(vals), np.sort(vals)[ int(0.05*N) ], np.sort(vals)[ int(0.95*N) ]
                    varDict[param],varDictLower[param],varDictUpper[param]=avg,lower,upper

                if key == 'sim':
                    y0s=[y01, 10., y02]
                else:
                    y0s=[y01, 0.1, y02]
                ys = solve_ivp(dydt,[0,ts[-1]],y0s,args=(varDict,),t_eval=ts,method='LSODA').y
                sim1,sim2=ys[0],ys[2] 
                ys = solve_ivp(dydt,[0,ts[-1]],y0s,args=(varDictLower,),t_eval=ts,method='LSODA').y
                sim1Lower,sim2Lower=ys[0],ys[2]  
                ys = solve_ivp(dydt,[0,ts[-1]],y0s,args=(varDictUpper,),t_eval=ts,method='LSODA').y
                sim1Upper,sim2Upper=ys[0],ys[2]  

                # normalization
                sim1=minmax_scaling(sim1)
                sim2=minmax_scaling(sim2)
                sim1Lower=minmax_scaling(sim1Lower)
                sim2Lower=minmax_scaling(sim2Lower)
                sim1Upper=minmax_scaling(sim1Upper)
                sim2Upper=minmax_scaling(sim2Upper)

                # rescale
                sim1=sim1*(channel1Max-channel1Min)+channel1Min
                sim2=sim2*(channel2Max-channel2Min)+channel2Min
                sim1Lower=sim1Lower*(channel1Max-channel1Min)+channel1Min
                sim2Lower=sim2Lower*(channel2Max-channel2Min)+channel2Min
                sim1Upper=sim1Upper*(channel1Max-channel1Min)+channel1Min
                sim2Upper=sim2Upper*(channel2Max-channel2Min)+channel2Min

                # area
                ax.fill_between(ts,sim1Lower,sim1Upper,alpha=0.5,color=cmap1(gradient[cell]), label='cell {}'.format(cell+1))
                axr.fill_between(ts,sim2Lower,sim2Upper,alpha=0.5,color=cmap2(gradient[cell]), )

                cell+=1

                if key=='expt':
                    loc=0
                else:
                    loc=5
                ax.legend(loc=loc)
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
            cmap1=None,
            cmap2=None,
            frameRate=None,
            ):

    rcParams['font.size']=17
    rcParams['figure.dpi']=300

    for key in data.keys():
        if key=='expt':
            rcParams['figure.figsize']=16,6
            nrows,ncols=2,4
        else:
            rcParams['figure.figsize']=16,5
            nrows,ncols=3,4

        n=nCells[key]
        df=infData[key]
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols)#, sharex='col', sharey='row')
        cell=0
        gradient=np.linspace(0,1,n)

        for row in range(nrows):
            for col in range(ncols):
                ax = axes[row, col]    
                axr=ax.twinx()
                ax.tick_params(axis='y', colors='forestgreen')
                axr.tick_params(axis='y', colors='firebrick')
                
                if key=='expt':
                    ax.set_ylim([-0.1, 1.2])
                    axr.set_ylim([-0.1, 1.2])
                
                if col == 0:  # Leftmost column
                    ax.set_ylabel('[Ca$^{2+}$]$_i$ (%)',color='forestgreen')    
                else:
                    ax.set_yticklabels([])
                if col == ncols-1:
                    axr.set_ylabel('[Ca$^{2+}$]$_{ER}$ (%)',color='firebrick')
                else:
                    axr.set_yticklabels([])
                if row == nrows - 1:  # Bottom row
                    ax.set_xlabel('time (s)')

                try:
                    channel1=data[key]['cyto'][cell]
                    channel2=data[key]['ER'][cell]
                except:
                    ax.set_visible(False) 
                    axr.set_visible(False) 
                    continue

                channel1=minmax_scaling(channel1)
                channel2=minmax_scaling(channel2)

                # raw data
                stepSize=len(channel1)
                ts=np.linspace(0,stepSize*frameRate[key],stepSize) 

                # run sma on experimental data
                if key == 'expt':
                    # sma
                    window1=window2=5
                    channel1=sma.sma(window1,channel1)
                    channel2=sma.sma(window2,channel2)
                    stepSize1=len(channel1)
                    ts=np.linspace(0,stepSize1*frameRate[key],stepSize1)
                    stepSize2=len(channel2)
                    ts2=np.linspace(0,stepSize2*frameRate[key],stepSize2)
                    # to ensure the two channels have the same time steps, we interpolate channel2 over the time steps of channel1
                    interp_func=interp1d(ts2,channel2,fill_value='extrapolate')
                    channel2=interp_func(ts)              

                y01=channel1[0]
                y02=channel2[0]

                channel1Min,channel1Max,channel2Min,channel2Max=np.min(channel1),np.max(channel1),np.min(channel2),np.max(channel2)
                
                for idx in range(nSample):
                    varDict={}
                    for i,param in enumerate(params):
                        vals=np.array([]) # array to combine all chains
                        for chain in range(nChains):  
                            trace=df.loc[df['chain'] == chain, "('posterior', '{}[{}]', {})".format(param,cell,cell)]
                            vals=np.append(vals,trace)
                        draw=np.random.choice(vals,replace=True)
                        varDict[param]=draw

                    if key == 'sim':
                        y0s=[y01, 10., y02]
                    else:
                        y0s=[y01, 0.1, y02]
                    ys = solve_ivp(dydt,[0,ts[-1]],y0s,args=(varDict,),t_eval=ts,method='LSODA').y
                    sim1,sim2=ys[0],ys[2] 

                    # normalization
                    sim1=minmax_scaling(sim1)
                    sim2=minmax_scaling(sim2)

                    # rescale
                    sim1=sim1*(channel1Max-channel1Min)+channel1Min
                    sim2=sim2*(channel2Max-channel2Min)+channel2Min

                    # area
                    ax.plot(ts,sim1,alpha=0.1,color=cmap1(gradient[cell]), label='cell {}'.format(cell+1))
                    axr.plot(ts,sim2,alpha=0.1,color=cmap2(gradient[cell]))

                    if idx==0:
                        if key=='expt':
                            loc=0
                        else:
                            loc=5
                        ax.legend(loc=loc)
                
                ax.plot(ts,channel1,lw=2,color='gray',alpha=0.5,linestyle='--')  
                axr.plot(ts,channel2,lw=2,color='gray',alpha=0.5,linestyle='--')  
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
            cmap1=None,
            cmap2=None,
            k_cols=None,
            ):
    
    rcParams['figure.dpi']=300
    rcParams['font.size']=15
    rcParams['figure.figsize']=9,3
    priors={}

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
                sns.kdeplot(vals,color=cmap1(gradient[cell]),fill=True).set(ylabel=None)

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
             cmap1=None,
             cmap2=None,
             k_cols=None,
            ):
    
    rcParams['figure.dpi']=300
    rcParams['font.size']=15

    for key in data.keys():
        df=infData[key]
        n=nCells[key]
        gradient=np.linspace(0,1,n)

        # hypers
        rcParams['figure.figsize']=10,5
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
        rcParams['figure.figsize']=10,2.5
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
                    ax.plot(xs,trace,color=cmap1(gradient[cell]),alpha=0.3)
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
            cmap1=None,
            cmap2=None,
            k_cols=None,
            ):
    
    rcParams['figure.dpi']=300
    rcParams['font.size']=15

    for key in data.keys(): 
        df=infData[key]
        n=nCells[key]
        gradient=np.linspace(0,1,n)

        rcParams['figure.figsize']=10,5
        fig=plt.figure()  
        ax=fig.add_subplot(1,4,1)
        ax.set_title('w',fontsize=25)
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
        
        # params
        rcParams['figure.figsize']=10,2.5
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
                    ax.plot(autocorr,color=cmap1(gradient[cell]),alpha=0.5)     
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
            cmap1=None,
            cmap2=None,
            frameRate=None,
            ):
    
    mses={'sim':{'cyto':[],'ER':[]},
      'expt':{'cyto':[],'ER':[]}}
    nIter=1000

    for key in data.keys():
        df=infData[key]
        n=nCells[key]

        for cell in range(n):
            mse1=[]
            mse2=[]

            channel1=data[key]['cyto'][cell]
            channel2=data[key]['ER'][cell]
            channel1=minmax_scaling(channel1)
            channel2=minmax_scaling(channel2)

            # raw data
            stepSize=len(channel1)
            ts=np.linspace(0,stepSize*frameRate[key],stepSize) 

            # run sma on experimental data
            if key == 'expt':
                # sma
                window1=window2=5
                channel1=sma.sma(window1,channel1)
                channel2=sma.sma(window2,channel2)
                stepSize1=len(channel1)
                ts=np.linspace(0,stepSize1*frameRate[key],stepSize1)
                stepSize2=len(channel2)
                ts2=np.linspace(0,stepSize2*frameRate[key],stepSize2)
                # to ensure the two channels have the same time steps, we interpolate channel2 over the time steps of channel1
                interp_func=interp1d(ts2,channel2,fill_value='extrapolate')
                channel2=interp_func(ts)              

            y01=channel1[0]
            y02=channel2[0]

            channel1Min,channel1Max,channel2Min,channel2Max=np.min(channel1),np.max(channel1),np.min(channel2),np.max(channel2)

            varDict={}

            for j in range(nIter):
                for i,param in enumerate(params):
                    vals=np.array([]) # array to combine all chains
                    for chain in range(nChains):  
                        trace=df.loc[df['chain'] == chain, "('posterior', '{}[{}]', {})".format(param,cell,cell)]
                        vals=np.append(vals,trace)         
                    draw=np.random.choice(vals,replace=True)
                    varDict[param]=draw

                if key == 'sim':
                    y0s=[y01, 10., y02]
                else:
                    y0s=[y01, 0.1, y02]
                ys = solve_ivp(dydt,[0,ts[-1]],y0s,args=(varDict,),t_eval=ts,method='LSODA').y
                sim1,sim2=ys[0],ys[2] 

                # normalization
                sim1=minmax_scaling(sim1)
                sim2=minmax_scaling(sim2)

                # rescale
                sim1=sim1*(channel1Max-channel1Min)+channel1Min
                sim2=sim2*(channel2Max-channel2Min)+channel2Min

                # MSE
                m1=np.mean((sim1-channel1)**2)
                m2=np.mean((sim2-channel2)**2)
                mse1.append(m1)
                mse2.append(m2)

            mses[key]['cyto'].append(mse1)
            mses[key]['ER'].append(mse2)
    
    rcParams['figure.dpi']=300
    rcParams['font.size']=8
    ylabels=['cyto','ER2']
    #bootstrapping
    nIter=500
    nSample=20

    for i,key1 in enumerate(mses.keys()):
        if key1=='sim':
            rcParams['figure.figsize']=4,4
        else:
            rcParams['figure.figsize']=3,4

        fig=plt.figure()
        n=nCells[key1]
        gradient=np.linspace(0,1,n)

        for j,key2 in enumerate(mses[key1].keys()):
            ax=fig.add_subplot(2,1,j+1)
            ax.set_xlabel('cell')
            ax.set_xticks(np.arange(n+1))
            ax.set_ylabel('MSE ({})'.format(ylabels[j]))
            if key1 == 'sim':
                ax.set_ylim([0.00005, 0.05])
                yloc=0.035
            else:
                ax.set_ylim([0.00005, 0.07])
                yloc=0.05

            vals=mses[key1][key2]
            parts=ax.violinplot(vals,showextrema=False)
            for idx,pc in enumerate(parts['bodies']):
                if key2=='cyto': 
                    pc.set_facecolor(cmap1(gradient[idx]))
                else:
                    pc.set_facecolor(cmap2(gradient[idx]))
                pc.set_alpha(0.3)

            for cell in range(n):
                ys=vals[cell]
                xs=np.random.normal(cell+1, 0.04, len(ys))
                if j == 0:
                    ax.scatter(xs,ys,s=1,color=cmap1(gradient[cell]),alpha=0.1)
                else:
                    ax.scatter(xs,ys,s=1,color=cmap2(gradient[cell]),alpha=0.1)

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
        cmap1=None,
        cmap2=None,
        k_cols=None,
        ):


        data,infData,nCells=loadData(exptFile=exptFile,simFile=simFile,frameRate=frameRate,
                                     nChains=nChains,K=K,model=model,infDataPath=infDataPath,
                                    reorderParam=reorderParam,paramList=paramList,)
    
    
        plotObs(data=data,nCells=nCells,frameRate=frameRate,cmap1=cmap1,cmap2=cmap2,)
    
    
        plotFits(data=data,nCells=nCells,infData=infData,params=params,frameRate=frameRate,
            fixedParams=fixedParams,K=K,nChains=nChains,cmap1=cmap1,cmap2=cmap2,model=model,)
        
        plotPPC(data=data,nCells=nCells,infData=infData,params=params,frameRate=frameRate,
            fixedParams=fixedParams,K=K,nChains=nChains,cmap1=cmap1,cmap2=cmap2,model=model,)
    
    
        plotPos(data=data,nCells=nCells,infData=infData,params=params,
            hypers_mu=hypers_mu,hypers_sigma=hypers_sigma,K=K,nChains=nChains,hyperLabels=hyperLabels,labels=labels,
            cmap1=cmap1,cmap2=cmap2,k_cols=k_cols,model=model,)
        
        
        plotTrace(data=data,nCells=nCells,infData=infData,params=params,K=K,nChains=nChains,
             hyperLabels=hyperLabels,sigmaLabels=sigmaLabels,
                  labels=labels,cmap1=cmap1,cmap2=cmap2,k_cols=k_cols,model=model,)
    
    
        plotCorr(data=data,nCells=nCells,infData=infData,params=params,K=K,nChains=nChains,
            hyperLabels=hyperLabels,sigmaLabels=sigmaLabels,
                 labels=labels,cmap1=cmap1,cmap2=cmap2,k_cols=k_cols,model=model,)
        
        
        plotJoint(data=data,nCells=nCells,infData=infData,params=params,
            hypers_mu=hypers_mu,hypers_sigma=hypers_sigma,K=K,nChains=nChains,hyperLabels=hyperLabels,labels=labels,
            k_cols=k_cols,model=model,)
        
        
        plotMSE(data=data,nCells=nCells,infData=infData,params=params,K=K,nChains=nChains,
                fixedParams=fixedParams,cmap1=cmap1,cmap2=cmap2,model=model,frameRate=frameRate,)
        

Ks=[2,3,4]

for k in Ks:

    run(
        exptFile='/home/xfang2/backup/faust_backup/automagikFitting/pymc3/cicr/expt/peaks_downsampled.csv',

        simFile='/home/xfang2/backup/faust_backup/automagikFitting/pymc3/cicr/sim/simData/trans_multiClass.pkl',

        infDataPath='/home/xfang2/backup/faust_backup/automagikFitting/pymc3/cicr',

        truthVals={
'kSC': [3.282861098239496, 2.285414583176121, 3.8596241810556586, 2.8124088623449723,2.143196686120475, 1.0723482647298106, 1.679732148309085, 1.800762611286055,3.7908909325080824, 2.7184168291764603, 3.4053324510734284, 1.9097836640458818], 
'kRyr': [0.017117645066539955, 0.023548651761230954, 0.023438353654869667, 0.017453905982330604,0.021621813646280765, 0.02115636776392014, 0.025284632768517543, 0.013812377787083038,0.004816891459821002, 0.0060589691875711504, 0.004602159771800009, 0.005337437653613972], 
'kcomp': [1.3040810567497516, 0.7169019119117946, 1.5990092884271527, 1.5579059658090963,0.4797353675370818, 0.4344030655861066, 0.5193421376470359, 0.5553438910956742,3.6285471437356334, 3.627562953376599, 3.518230375010903, 2.9267450550913954], 
},
        
        cmap1=matplotlib.colormaps['viridis'],
        
        cmap2=matplotlib.colormaps['autumn'],
        
        k_cols=['lightskyblue','deepskyblue','royalblue','tab:blue'],

        hypers_mu={
    'sim':{'mu_kSC': 2.0, 'mu_kRyr': 2.0e-2, 'mu_kcomp': 1.0},
    'expt':{'mu_kSC': 3.0, 'mu_kRyr': 2.0e-2, 'mu_kcomp': 2.0e-1}},

        hypers_sigma={
    'sim':{'mu_kSC': 1.0e-1, 'mu_kRyr': 1.0e-1, 'mu_kcomp': 1.0e-1},
    'expt':{'mu_kSC': 1.0e-1, 'mu_kRyr': 1.0e-1, 'mu_kcomp': 1.0e-1}},

        hyperParams=['mu_kSC','mu_kRyr','mu_kcomp'],

        params=['kSC','kRyr','kcomp'],

        paramList=['w','mu_kSC','mu_kRyr','mu_kcomp','sigma_kSC','sigma_kRyr','sigma_kcomp'],

        hyperLabels=['$\mu_{k_{SERCA}}$', '$\mu_{k_{Ryr}}$', '$\mu_{k_{shuttle}}$'],
        
        sigmaLabels=['$\sigma_{k_{SERCA}}$', '$\sigma_{k_{Ryr}}$', '$\sigma_{k_{shuttle}}$'],

        labels=['k$_{SERCA}$', 'k$_{Ryr}$', 'k$_{shuttle}$'],

        frameRate={'sim':0.05,'expt':3.0e-2},

        nChains=5,

        K=k,

        model=1,

        reorderParam='mu_kcomp',  
    )
        
        
        
    
