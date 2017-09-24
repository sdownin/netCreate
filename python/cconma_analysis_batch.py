# -*- coding: utf-8 -*-
"""
Created on Sat May 30 01:35:46 2015

@author: Stephen

CCONMA CUSTOMER NETWORK CREATION 
PRELIMINARY ANALYSIS

"""
from __future__ import division

import os
os.chdir('C:\\Users\\T430\\Google Drive\\PhD\\Dissertation\\3. network analysis\\data\\netcreate\\python')
import netcreate_batch as nc
os.chdir('C:\\Users\\T430\\Google Drive\\PhD\\Dissertation\\3. network analysis\\data')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import datetime as dt
import networkx as nx
from time import time
from sklearn.decomposition import PCA, NMF
from argparse import ArgumentParser
from sklearn.metrics import roc_curve, auc, confusion_matrix
from scipy.sparse import csr_matrix

## input arguments
par = ArgumentParser(description="Network Creation Inference")
par.add_argument('n', nargs='?', type=int, default=700, help="customer sample size [default 700]")
par.add_argument('--tol', type=float, default=.001, help="tolerance above which SIF/max(SIF) weight predicts a tie (y_ij=1) [default 0.001]")
args = par.parse_args()
# sample size
n = args.n
# tolerance for tie prediction (classification)
tol = args.tol

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
#  - if n < m (num recommended customers), then use only those m
#  - if n >= m (num recommended customers), then use m recommended + (n-m) sample of non-recommended
#
#----------------------------------------------------

############
#
##n = len(dfm.mem_no.unique())  # use all who reviewed and bought a product
#n = 1000
#
############

np.random.seed(111)

#samp_mem = np.int32( np.random.choice(dfm.mem_no.unique(), size=n_mem, replace=False) )
#samp_rec = np.int32( np.random.choice(dfm.recommender.unique(), size=n_rec, replace=False) )
#memsamp = list(pd.Series(np.int32(np.concatenate((samp_mem, samp_rec )))).unique())
# members who also recommended
mem_rec = list(dfm.mem_no.loc[dfm.mem_no.isin(dfm.recommender)].unique())

n_no_rec = n - len(mem_rec)
if n_no_rec > 0:
    samp = np.int32( np.random.choice(dfm.mem_no.loc[~dfm.mem_no.isin(mem_rec)].unique(), size=n_no_rec, replace=False) )
    memsamp = list(pd.Series(np.int32(np.concatenate((samp, mem_rec )))).unique())
    len_rec = len(mem_rec)
    len_no_rec = n_no_rec
else:
    memsamp = list(np.int32( np.random.choice(mem_rec, size=n, replace=False) ))
    len_rec = n
    len_no_rec = 0

dfsub = dfm.loc[dfm.mem_no.isin(memsamp), : ].copy()

print("Running netcreate with sample size %d (%d recommenders, %d not)" % (n, len_rec, len_no_rec))



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
dfsim = df1s.loc[df1s.mem_no.isin(memsamp),['mem_no','recommender','genderint','marriageint','agecatint']].merge(dfqtyrev, how='right', on='mem_no')
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
b.build_sim_tensor( dfsim )

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
reg = 2

#time0 = time()
# decompose tensor
b.decompose_tensor(rank=rank, init='nvecs', lambda_A=reg, lambda_R=reg, compute_fit=False)


#averge degree of k = 5
minEdges = int(b.X[0].shape[0]*4)

## create network via sampling methods specified
b.net_create(minEdges=minEdges, deterministic=True, Bernoulli=True, 
             plotting=False, color_threshold=0.35)
#print(str(round(time()-time0,3)))



#--------------- LOW RANK---HIGH REGULATION ----------------------------
rank = int(np.ceil(c.X[0].shape[0]* 0.7 ))   # rank ~ 90% of number of people
reg = 20

# decompose tensor
c.decompose_tensor(rank=rank, init='nvecs', lambda_A=reg, lambda_R=reg, compute_fit=False)

## create network via sampling methods specified
c.net_create(minEdges=minEdges, deterministic=True, Bernoulli=True, 
             plotting=False, color_threshold=0.35)


print(b)
nnz = np.sum([ b.X[k].getnnz() for k in range(len(b.X)) ])
n = b.AAT.shape[0]
r = len(b.X)
percentnnz = '< 0.01' if 100*nnz/(n**2 * r) < 0.0001 else  round(100 * nnz/(n**2 * r) , 4)
print('\nSimilarity tensor:')
print('Dimensions: [ %s x %s x %s ]' % (n, n, r))
print('Nonzero elements: %s (%s %s)' % (nnz, percentnnz, '%' ))

# save netCreate object
pickle.dump( b , open("nc_regpred_n670_obj_high_rank95_low_reg4_colthresh45.p","wb" ) )
pickle.dump( c , open("nc_regpred_n670_obj_low_rank70_high_reg15_colthresh45.p","wb" ) )



##plot the three different probability distributions
df = pd.DataFrame({'HrankLreg':b.pred_rank['Bernoulli'][['prob']].reset_index()['prob'],
                   'LrankHreg':c.pred_rank['Bernoulli'][['prob']].reset_index()['prob'] })

me = int( np.ceil(c.X[0].shape[0] * (c.X[0].shape[0] - 1) / 50 ) )
df.iloc[:,:].sort_values('HrankLreg').plot(marker='^',markevery=me,title="Network 'Next Tie Probability'\nby tensor decomp hyperparams")
plt.savefig("next_tie_prob.png",dpi=200)
#plt.show()




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
#plt.show()
 
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
#plt.show()

nKeep = 4
revtrans = revpca.transform( X )
dfpcarev = pd.DataFrame(revtrans[: , 0:nKeep])
dfpcarev = pd.concat( (revwide.iloc[:,0], dfpcarev), axis=1)
dfpcarev = dfpcarev.iloc[1:N,:]
dfpcarev.columns = ['mem_no','revPC0','revPC1','revPC2','revPC3']


#
# Combine  PCA results
dfpca = dfpcarev.merge(dfpcaqty, on=['mem_no'], how='outer')
dfreg = df1s[['mem_no','age','gender', 'marriage','recommender' ]].merge(dfpca, on='mem_no', how='right')
#------------------------------------------------------







#-----------------------------------------------------
#
# Make Social Influence Factor Matrix
# Ties strengths by other members' purchases
#
#_----------------------------------------------------


# compute the weights matrix
N,M = qtywide.shape
W = b.SIF.dot( qtywide.iloc[:,1:M] ) 

# back to pandas dataframe and add the mem_no to join with the regression data
Wdf = pd.DataFrame(W)
Wdf = pd.concat(( dfsim.iloc[:N,0], Wdf), axis=1)
prefs = [revwide.columns[x].split("-")[0] for x in range(revwide.shape[1])]
prefs[0] = 'mem_no'
Wdf.columns = prefs
Wlong = pd.melt(Wdf, id_vars=['mem_no'])
Wlong.rename(columns={'value':'netWeight', 'variable':'pref'}, inplace=True)

# combine final multi-record regression  dataframe
dfm.drop(labels=['genderint','marriageint','agecatint'], axis=1, inplace=True)
dfregall = dfm.merge(dfreg,on=['mem_no','age','gender','marriage','recommender'],how='outer')
dfregall = Wlong.merge(dfregall, on=['mem_no','pref'],how='inner')
dfregall.to_csv("dfregall.csv",sep=",",index=False)



#-----------------------------------------------------
#
# Network Prediction 
# AUC of SIF weights vs actual recommended ties. 
#
#_----------------------------------------------------
print("Assessing network prediction accuracy...")
## save network structure
# edgelist
el = dfregall.loc[:,['recommender','mem_no']].drop_duplicates()
el.to_csv('graph_el.csv', index=False)

# vertex attributes df
# all memnos in connected graph : mem_no and recommender
#memnos = list(pd.Series(np.int32(np.concatenate((dfregall.mem_no.values, dfregall.recommender.values )))).unique())
# only memnos
vertices = df1.loc[df1.mem_no.isin(el.mem_no),:].copy()
vertices.to_csv('graph_vertices.csv', index=False)

# true network
g = nx.Graph()
# add nodes with colors of group
for n in vertices.mem_no: 
    g.add_node(n)
# # add edges with weight of theta (probability the link exists)
for e in el.index:
    edge = el.loc[e,:]
    if edge.recommender in g.nodes:
        print("adding edge %d --> %d" % (edge.recommender, edge.mem_no))
        g.add_edge(int(edge.recommender),int(edge.mem_no))
        
# TRUE sparse adjacency matrix
net_sparse = nx.adjacency_matrix(g)
net = csr_matrix(net_sparse).toarray()

#
# PREDICTED
#
print('Searching for optimum tie classification tolerance...')
model = b
pred =  model.SIF / np.max(model.SIF)
plt.figure()
plt.plot( np.sort( pred[np.triu_indices(pred.shape[0])] ) )
#plt.show()
tols = [];
aucs = [];
for tol_i in np.arange(.01,1,.01):
    pred =  model.SIF / np.max(model.SIF)
    pred[ pred >= tol_i ] = 1
    pred[ pred < tol_i  ] = 0
    
#    # test vals
#    rec_indices_1 = np.where(net == 1)
#    rows = [];
#    cols = [];
#    for i,j in zip(rec_indices_1[0],rec_indices_1[1]):
#        if i < j:
#            rows.append(i)
#            cols.append(j)
#    rec_indices_0 = np.where(net == 0)
#    for k,tup in enumerate(zip(rec_indices_0[0],rec_indices_0[1])):
#        if tup[0] < tup[1]:   # and k <= 100*len(rec_indices_1[0])
#            rows.append(tup[0])
#            cols.append(tup[1])
#    rec_indices_triu = (rows, cols)
    rec_indices_triu = np.triu_indices(pred.shape[0])
    
    y_true = net[  rec_indices_triu ]
    y_pred = pred[ rec_indices_triu ]
    
    # measure accuracy
    fpr, tpr, thresh = roc_curve(y_true, y_pred)
    auc_i = auc(fpr, tpr)
    aucs.append( auc_i )
    tols.append( tol_i )
    print('tol: %.3f | auc: %.3f' % (tol_i, auc_i) )


optim_tol = tols[np.where(aucs == max(aucs))[0][0] ] 
print('Optimum tolerance = %.3f' % optim_tol)

plt.figure()
pd.DataFrame([{'tol':i,'auc':j} for i,j in zip(tols,aucs) ]).plot(x='tol',y='auc', title='Prediction Accuracy by Tie Classification Tolerance')
plt.axvline(x= optim_tol , color='k')
plt.axhline(y=.5 , color='k')
plt.savefig('predition_accuracy_auc_by_tol',  dpi=200  )
#plt.show()

print('y_true len: %d, y_pred len: %d' %(y_true.shape[0], y_pred.shape[0]))

tol = optim_tol

model = b
pred =  model.SIF / np.max(model.SIF)
pred[ pred >= tol ] = 1
pred[ pred < tol  ] = 0

## test vals
rec_indices_triu = np.triu_indices(pred.shape[0])

y_true = net[  rec_indices_triu ]
y_pred = pred[ rec_indices_triu ]

# measure accuracy
fpr, tpr, thresh = roc_curve(y_true, y_pred)
roc_auc = auc(fpr, tpr)
conf_mat = confusion_matrix(y_true, y_pred)
print(conf_mat)

# plot AUC curve
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Netcreate Prediction (%s edges among %s nodes)' % (len(g.edges),len(g.nodes)))
plt.legend(loc="lower right")
#
plt.savefig('netcreate_ROC_%s.png' % str(time0), dpi=200)
#plt.show()


timeout = time() - time0    
if timeout <= 60:
    print('\nElapsed time: %s seconds' % ( round(timeout,3) ))
elif timeout <= 3600: 
    print('\nElapsed time: %s minutes' % ( round(timeout/60,3) ))
else:
    print('\nElapsed time: %s hours' % ( round(timeout/3600,3) ))
  
print('\nScript completed successfully.')

