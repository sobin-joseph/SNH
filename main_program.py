import time
import os
import numpy as np
import pandas as pd
import Analysis.SNH_univariate as univariate
import Analysis.SNH_bivariate as bivariate
nNeurons=60
delta=1
alpha=0.3
beta=3
mu_t=0.5
def Univariate():

    kernelType = 0
    if(kernelType == 0):
        tseries = univariate.simu_univariate(alpha,beta,mu_t,6000)
    elif(kernelType == 1):
        tseries = univariate.simu_uniRectangle(alpha,delta,beta,mu_t,60000)
    
    t = np.array(tseries)
    t=t.reshape(-1)

    goodInitialization = False
    x = univariate.initializeParams(nNeurons)
    #if(goodInitialization):
    #    nNeurons=10
    #    x = initializeParams(nNeurons)
    #    betas = np.array([-0.29404953, -0.7008895 ,  0.6594665 ,  0.9012316 ,  1.021717  ,
    #          1.1006424 , -0.6777213 ,  0.7278789 ,  0.12091339, -0.10012847]).reshape(-1,1)
    #    beta0 = np.array([ 0.0000000e+00,  0.0000000e+00,  2.4713385e-03,  7.9669163e-04,
    #          4.4815044e-04,  2.1983573e-04,  0.0000000e+00,  2.0738167e-03,
    #         -2.3580924e-01,  0.0000000e+00]).reshape(-1,1)
    #
    #    alphas = np.array([[ 0.13526493],
    #        [ 0.3087886 ],
    #        [-0.8584821 ],
    #        [-0.09322891+1],
    #        [-0.7955883 ],
    #        [-0.7802333 ],
    #        [-0.2486592 +1],
    #        [-0.9317662 ],
    #        [ 0.32983962],
    #        [-0.24496266]])
    #    alpha0 = 0.69780564
    #    A[0]=alphas
    #    A[1]=alpha0
    #    B[0]=betas+(np.random.uniform(-0.1,0.1,nNeurons)).reshape(-1,1)
    #    B[1]=beta0+(np.random.uniform(-0.1,0.1,nNeurons)).reshape(-1,1)

    univariate.inflectionPoints()
    #univariate.plotKernels()
           


    start_time = time.time()

    mu=univariate.sgdNeuralHawkes(30,0.01,x,t)
    print("Actual log likelihood values are")
    if(kernelType == 1):     
        print(univariate.loglikelihood_rect(np.array([alpha,beta,delta,mu_t])),univariate.nnLoglikelihood(mu_t))
    else:
        print(univariate.loglikelihood(np.array([alpha,beta,mu_t])),univariate.nnLoglikelihood(mu_t))

    print("--- %s seconds ---" % (time.time() - start_time))

def Multivariate():
   
    url = "https://raw.githubusercontent.com/lekhapriya/SNH/master/Combined.csv"
    dataset = pd.read_csv(url, names=['Timestamp', 'Price','Volume','Buyer ID','Seller ID','Buyer is market maker'])
    dataset = dataset.sort_values(dataset.columns[0])
    buy_df = dataset.loc[~dataset.iloc[:,5]]
    sell_df = dataset.loc[dataset.iloc[:,5]]
    buy = buy_df.drop_duplicates(['Buyer ID'])
    sell = sell_df.drop_duplicates(['Seller ID'])

    #ms to seconds
    utc_sell = (sell.iloc[:,0].values)/1000
    utc_buy = (buy.iloc[:,0].values)/1000
    intial_value=min(utc_sell[0],utc_buy[0])
    #convert utc to seconds since t0
    t_sell = []
    for i in range(0,len(utc_sell)) :
        t_sell.append(utc_sell[i]-intial_value)
    
    t_buy = []
    for i in range(0,len(utc_buy)) :
        t_buy.append(utc_buy[i]-intial_value)
    
    print('N =',len(t_sell)+len(t_buy))
                            
    nNeurons = 100
    bivariate.initializeParams(nNeurons)
    bivariate.inflectionPoints()
    t = list([t_sell,t_buy])
   
   
    SGD = bivariate.sgdNeuralHawkesBiVariate(30,0.01,t)


Multivariate()
    
