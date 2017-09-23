# -*- coding: utf-8 -*-
"""
Created on Sat May 30 01:35:46 2015

@author: Stephen

CCONMA CUSTOMER NETWORK CREATION 
PRELIMINARY ANALYSIS

"""
import os
os.chdir('C:\\Users\\T430\\Google Drive\\PhD\\Dissertation\\3. network analysis\\data')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import netcreate_batch as nc
import pickle
import datetime as dt
from time import time
from sklearn.decomposition import PCA, NMF

# time
time0 = time()

#-----------------------------------------------------
#
# 1. Subset data and prepare dataframe to build 
#    similarity tensor
#
#-----------------------------------------------------

# all MEMBERS who ordered at least one product
df1 = pd.read_csv("df1_mem.csv")

# all PRODUCT REVIEWS
df2 = pd.read_csv("df2_rev.csv")

# product ORDER QUANTITY by member
df3 = pd.read_csv("df3_qty.csv")

## REMOVE mem_no=0
df1 = df1.loc[df1.mem_no != 0,:].copy()
df2 = df2.loc[df2.mem_no != 0,:].copy()
df3 = df3.loc[df3.mem_no != 0,:].copy()

# subset orders by the customers who reviewed products
revmem = df2.mem_no.loc[df2.mem_no.isin(df1.mem_no.values)].unique()
df3s = df3.loc[df3.mem_no.isin(revmem), : ].copy()
df1s = df1.loc[df1.mem_no.isin(revmem), : ].copy()

# keep only ordered products that are also reviewed
dfm = df3s.merge(df2, how='inner', on=['pref','mem_no','pcode'])
dfm = dfm.merge(df1s, how='outer', on=['mem_no'])

# make purchases quantities yearly average and then round
years = 22/12
dfm.qty = np.round(dfm.qty / years, 0)

#----------------------------------------------------
#
# SAMPLE of n customers
#
#----------------------------------------------------

###########

#n = len(dfm.mem_no.unique())  # use all who reviewed and bought a product
n = 200

###########

np.random.seed(111)
memsamp = np.int32( np.random.choice(dfm.mem_no.unique(), size=n, replace=False) )
dfsub = dfm.loc[dfm.mem_no.isin(memsamp), : ].copy()

# CAST QUANTITY WIDE---------------------------------
qtywide = dfsub.pivot_table(index=['mem_no'],columns='pcode',values='qty',
                         aggfunc= np.sum, fill_value = np.nan )
qtywide.reset_index(inplace=True)


qcols = [ '%s_QTY' % (x) for x in qtywide.columns[1:qtywide.shape[1]]]
qcols.insert(0,'mem_no')
qtywide.columns = qcols
#  round to integers
M = qtywide.shape[1]
qtywide.iloc[:,1:M] = np.round(qtywide.iloc[:,1:M],0)


# CAST REVIEWS WIDE-----------------------------------
revwide = dfsub.pivot_table(index=['mem_no'],columns='pcode',values='point',
                         aggfunc = np.mean, fill_value = np.nan)
revwide.reset_index(inplace=True)

rcols = [ '%s_REV' % (x) for x in revwide.columns[1:revwide.shape[1]]]
rcols.insert(0,'mem_no')
revwide.columns = rcols

# round to integers;  
# first multiple review mean by 2 to allow separation after rounding (2,10) instead of (1,5)
M = revwide.shape[1]
revwide.iloc[:,1:M] = np.round(2* revwide.iloc[:,1:M],0)


# join MEMBERS, REVIEWS, QUANTITIES integers as similarity dataframe
# INPUTE FOR BUILDING SIMILARITY TENSOR
dfqtyrev = qtywide.merge(revwide, how='outer', on=['mem_no'])
dfsim = df1s.loc[:,['mem_no','recommender','genderint','marriageint','agecatint']].merge(dfqtyrev, how='right', on='mem_no')
## replace
dfsim.recommender = dfsim.recommender.apply(lambda x: 0 if pd.isnull(x) else int(x))
#----------------------------------------------------




#----------------------------------------------------
#
#
# 2. build sim tensor from data subset in step 1
#    decompose using RESCAL_ALS
#    then create network and plot
#
#
#_---------------------------------------------------


#----------------------------------------------
#
# 2.A  Create new netCreate objects by BUILDING TENSOR
#
#----------------------------------------------

# 1. create object without premade similarity tensor
b = nc.netCreate()

# 2. build the similarity tensor from the data subset

#nmem = 3000
#np.random.seed(111)
#sample = np.random.choice(np.arange(dfsub.shape[0]), nmem, replace=False)

# bulid sim tensor
N,M = dfsim.shape
b.build_sim_tensor( dfsim.iloc[1:N,:] , offset=1)

## reuse same similarity tensor build in object b
c = nc.netCreate(b.X)

import gc
gc.collect()
#-----------------------------------------------
# 2.B  reload already saved TENSOR and use to create netCreate object
#-----------------------------------------------

#b = pickle.load( open( "netcreate_n3_obj_high_rank_low_reg.p", "rb" ) )
#c = pickle.load( open( "netcreate_n3_obj_low_rank_high_reg.p", "rb" ) )


#-------------------------------------
## Compute R, A, AAT, AATnn by decomposing tensor X using RESCAL_ALS
#-------------------------------------

#--------------- HIGH RANK---LOW REGULATION ---------------------------
rank = int( b.X[0].shape[0]* 0.99 )   # rank ~ 95% of number of people
reg = 5

#time0 = time()
# decompose tensor
b.decompose_tensor(rank=rank, init='nvecs', lambda_A=reg, lambda_R=reg, compute_fit=False)


#averge degree of k = 5
minEdges = int(b.X[0].shape[0]*4)

## create network via sampling methods specified
b.net_create(minEdges=minEdges, deterministic=True, Bernoulli=False, 
             plotting=False, color_threshold=0.35)
#print(str(round(time()-time0,3)))



#--------------- LOW RANK---HIGH REGULATION ----------------------------
rank = int(np.ceil(c.X[0].shape[0]* 0.7 ))   # rank ~ 90% of number of people
reg = 20

# decompose tensor
c.decompose_tensor(rank=rank, init='nvecs', lambda_A=reg, lambda_R=reg, compute_fit=False)

## create network via sampling methods specified
c.net_create(minEdges=minEdges, deterministic=True, Bernoulli=False, 
             plotting=False, color_threshold=0.35)


print(b)
nnz = np.sum([ b.X[k].getnnz() for k in range(len(b.X)) ])
n = b.AAT.shape[0]
r = len(b.X)
percentnnz = round( nnz/(n**2 * r) , 5) * 100
print('\nSimilarity tensor:')
print('Dimensions: [ %s x %s x %s ]' % (n, n, r))
print('Nonzero elements: %s (%s per cent)' % (nnz, percentnnz ))

# save netCreate object
pickle.dump( b , open("nc_regpred_n670_obj_high_rank95_low_reg4_colthresh45.p","wb" ) )
pickle.dump( c , open("nc_regpred_n670_obj_low_rank70_high_reg15_colthresh45.p","wb" ) )



##plot the three different probability distributions
df = pd.DataFrame({'HrankLreg':b.pred_rank['Bernoulli'][['prob']].reset_index()['prob'],
                   'LrankHreg':c.pred_rank['Bernoulli'][['prob']].reset_index()['prob'] })

me = int( np.ceil(c.X[0].shape[0] * (c.X[0].shape[0] - 1) / 50 ) )
df.iloc[:,:].sort().plot(marker='^',markevery=me,title="Network 'Next Tie Probability'\nby tensor decomp hyperparams")
plt.savefig("next_tie_prob.png",dpi=200)






#-------------------------------------------------------
#
# Dimensionality reduction on main regression behavior predictors
# PCA
#
#------------------------------------------------------

#----------------------------------------------------
# Prepare data set for PCA
# quantity missing assigned zero value
qtywide.replace(np.nan, 0, inplace=True)

# impute missing review values of product pref review by all customer who have reviewed it
revwide = revwide.fillna(revwide.mean(axis=0))

#-------------------------------------------
# QUANTITY PCA 
#------------------------------------------

nComps = 10
N, M = qtywide.shape
X = qtywide.iloc[:,1:M].copy()
qtypca = PCA(n_components = nComps)
qtypca.fit(X)
plt.figure()
plt.plot(np.arange(1,nComps+1),qtypca.explained_variance_ratio_, marker='^')
plt.title('Purchases Quantity: Explained Variance')
plt.xlabel('Principle Components')
plt.savefig("qty_pca_5.png",dpi=200)
 
nKeep = 4
qtytrans = qtypca.transform( X )
dfpcaqty = pd.DataFrame( qtytrans[: , 0:nKeep] )
dfpcaqty = pd.concat( (qtywide.iloc[:,0], dfpcaqty), axis=1)
dfpcaqty = dfpcaqty.iloc[1:N,:]  # ovservation mem_no 0
dfpcaqty.columns = ['mem_no','qtyPC0','qtyPC1', 'qtyPC2', 'qtyPC3']

#----------------------------------------------
# REVIEWS PCA 
#----------------------------------------------

nComps = 10
N, M = revwide.shape
X = revwide.iloc[:,1:M]
revpca = PCA(n_components = nComps)
revpca.fit( X )
plt.figure()
plt.plot(np.arange(1,nComps+1),revpca.explained_variance_ratio_, marker='^')
plt.title('Product Reviews: Explained Variance')
plt.xlabel('Principle Components')
plt.savefig("qty_pca_10.png",dpi=200)

nKeep = 4
revtrans = revpca.transform( X )
dfpcarev = pd.DataFrame(revtrans[: , 0:nKeep])
dfpcarev = pd.concat( (revwide.iloc[:,0], dfpcarev), axis=1)
dfpcarev = dfpcarev.iloc[1:N,:]
dfpcarev.columns = ['mem_no','revPC0','revPC1','revPC2','revPC3']


#
# Combine  PCA results
dfpca = dfpcarev.merge(dfpcaqty, on=['mem_no'], how='outer')
dfreg = df1s[['mem_no','age','gender', 'marriage' ]].merge(dfpca, on='mem_no', how='right')
#------------------------------------------------------







#-----------------------------------------------------
#
# Make Social Influence Factor Matrix
# Ties strengths by other members' purchases
#
#_----------------------------------------------------


# compute the weights matrix
N,M = qtywide.shape
W = b.SIF.dot( qtywide.iloc[1:N,1:M] ) 

# back to pandas dataframe and add the mem_no to join with the regression data
Wdf = pd.DataFrame(W)
Wdf = pd.concat(( dfsim.iloc[1:N,0], Wdf), axis=1)
prefs = [revwide.columns[x].split("_")[0] for x in range(revwide.shape[1])]
prefs[0] = 'mem_no'
Wdf.columns = prefs
Wlong = pd.melt(Wdf, id_vars=['mem_no'])
Wlong.rename(columns={'value':'netWeight', 'variable':'pref'}, inplace=True)

# combine final multi-record regression  dataframe
dfm.drop(labels=['genderint','marriageint','agecatint'], axis=1, inplace=True)
dfregall = dfm.merge(dfreg,on=['mem_no','age','gender','marriage'],how='outer')
dfregall = Wlong.merge(dfregall, on=['mem_no','pref'],how='inner')
dfregall.to_csv("dfregall.csv",delimiter=",",index=False)

timeout = time() - time0       
if timeout <= 3600: 
    print('\nElapsed time: %s minutes' % ( round(timeout/60,3) ))
else:
    print('\nElapsed time: %s hours' % ( round(timeout/3600,3) ))
  
print('\nScript completed successfully.')

