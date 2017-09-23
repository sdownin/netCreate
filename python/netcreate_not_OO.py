# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 12:08:48 2015

@author: Stephen
"""
import numpy as np
import matplotlib.pyplot as plt
import logging
import pandas as pd
import networkx as nx
import sktensor as st
from rescal import rescal_als
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform


def heatmap(dm):
    """ Input: data matrix;   
    Return: {'ordered' : D, 'rorder' : Z1['leaves'], 
    'corder' : Z2['leaves'], 'group':Z1['color_list']}
    """
    #from scipy.cluster.hierarchy import linkage, dendrogram
    #from scipy.spatial.distance import pdist, squareform
    #import matplotlib.pyplot as plt
    
    D1 = squareform(pdist(dm, metric='euclidean'))
    D2 = squareform(pdist(dm.T, metric='euclidean'))
    
    f = plt.figure(figsize=(8, 8))

    # add first dendrogram
    ax1 = f.add_axes([0.09, 0.1, 0.2, 0.6])
    Y = linkage(D1, method='complete')
    Z1 = dendrogram(Y, orientation='right')
    ax1.set_xticks([])
    ax1.set_yticks([])

    # add second dendrogram
    ax2 = f.add_axes([0.3, 0.71, 0.6, 0.2])
    Y = linkage(D2, method='complete')
    Z2 = dendrogram(Y)
    ax2.set_xticks([])
    ax2.set_yticks([])

    # add matrix plot
    axmatrix = f.add_axes([0.3, 0.1, 0.6, 0.6])
    idx1 = Z1['leaves']
    idx2 = Z2['leaves']
    D = D1[idx1, :]
    D = D[:, idx2]
    im = axmatrix.matshow(D, aspect='auto', origin='lower', cmap='hot')
    axmatrix.set_xticks([])
    axmatrix.set_yticks([])
    
    return {'ordered' : D, 'rorder' : Z1['leaves'], 'corder' : Z2['leaves'], 'group':Z1['color_list'], 'linkage':Y}

	
def triangleToVec(mat):
    """transform lower triangular matrix row-wise to numpy vector"""
    #import numpy as np
    n = np.shape(mat)[0]
    m = np.shape(mat)[1]
    qlist = []
    for i in np.arange(n):
        for j in np.arange(m):
            if i > j:
                qlist.append(mat[i,j])
				
    return np.asarray(qlist)


def vecToTriangle(vec):
    """transform vector to lower triangular matrix
    Input: one-dimensional numpy ndarray or list
    Output: square numpy ndarray
    """
    #import numpy as np
    E = len(vec)
    n = 0
    while E > 0:
        n += 1
        E = E - (n-1)
    M = np.zeros((n,n))
    # use numpy's lower triangular indices to populate matrix with vec elements
    M[np.tril_indices(n, -1)] = vec
	
    return M


def sampleLinks(q,edgesToDraw=1,draws=1):
    """sample links in lower triangular adjacency matrix from link distribution
    inputs: q vector of multinomial probabilities;
    if draws >1 returns mean links sampled for each element;
    outputs: lower triangular matrix
    """
    #import numpy as np
    #qvec = triangleToVec(q)
    qvec = q
    y = np.random.multinomial(edgesToDraw, qvec, draws)
    outvec = np.asarray([np.mean([y[draw][item] for draw in np.arange(draws)]) for item in np.arange(len(qvec))])
    outvec = np.ceil(outvec)
    return outvec


def netCreate(X, r, minEdges, sampleMethod='deterministic',
              rescal_lambda_A=10, rescal_lambda_R=10, 
              plotting=True, layout='spring', graphScale=1.0): 
    """Wrapper for sktensor.rescal_als. Create network by given 
    sampleMethod from hierarchical clustering of RESCALA_ALS 
    tensor factorization of singular values matrix A(A^T);
    
    Input: X is list of sktensor.csr_matrices [X[k] for k in 
    relationships],
    each X_k is frontal slide of adjacency tensor 
    (ie, adjacency matrix of one relationship type);
    
    Return: {'cluster':hm, 'graph':g, 'linkage':hm['linkage'], 
    'theta':theta, 'A':A, 'Z':Z}
    """
    #import logging
    #from rescal import rescal_als
    #import numpy as np
    #import pandas as pd
    #import networkx as nx
    #import matplotlib.pyplot as plt
    #from scipy.spatial.distance import pdist, squareform

    # Set logging to INFO to see RESCAL information
    logging.basicConfig(level=logging.INFO)

    # Decompose tensor using RESCAL-ALS
    A, R, fit, itr, exectimes = rescal_als(X, r, init='nvecs', lambda_A=rescal_lambda_A, lambda_R=rescal_lambda_R)

    # make the AAT matrix
    AAT = np.dot(A, A.T)

    # heatmap
    hm = heatmap(AAT)
    plt.suptitle(r'A(A^T) HAC for Induced Rank = %s, $\lambda_{A}$ = %s, $\lambda_{R}$ = %s '%(r,rescal_lambda_A,rescal_lambda_R), fontweight='bold', fontsize=14)

    # remove zeros
    AATnn = AAT
    AATnn[AATnn < 0] = 0
    # remove upper triangle
    AATnn = np.tril(AATnn, k= -1)  # k = -1 to keep only below diagonal

    # reproducibility
    np.random.seed(1)

    # network sampling
    if sampleMethod == 'Bernoulli':
        # shortcut: normalize by largest AAT (non-negative) value for separate bernoulli draws
        # instead of summing over all AAT for multinomial draw, which is harder to 
        # transfer back and forth between vector and triangular matrix
        theta = AATnn / AATnn.max()
        # NETWORK SAMPLE ALGORITHM:
        # random sample ties in network adjacency matrix
        # one-element-at-a-time Bernoulli shortcut
        # instead of multinomial sample of entire adjacency
        n = np.shape(theta)[0]
        m = np.shape(theta)[1]
        Z = np.zeros((n,m))
        # use dependent row,col permutations to randomly select
        # elements ij to sample after first full pass through matrix
        while np.sum(Z) < minEdges:
            shuffledRows = np.arange(1,n)  #up to n rows
            np.random.shuffle(shuffledRows)
            # first shuffle rows
            for i in shuffledRows: 
                # for given row shuffle use lower triangle columns j in that row i
                shuffledCols = np.arange(i) #up to (i-1) cols, ie, lower triangle
                np.random.shuffle(shuffledCols)
                for j in shuffledCols:
                    if Z[i,j] < 1:
                        Z[i,j] = np.random.binomial(n=1, p=theta[i,j], size=1)
                    if np.sum(Z) >= minEdges:
                        break
                if np.sum(Z) >= minEdges:
                    break
                
    elif sampleMethod == 'multinomial':
        # NETWORK SAMPLING ALGORITHM:
        # problem: doesn't sufficiently cluster the resulting network
        draws = int(np.ceil(minEdges*1.2))
        dist = pdist(A)   # what matrix to use:  pdist(A) or just tril(AAT) directly?
        invdist = dist
        invdist[invdist != 0] = 1/invdist[invdist!=0]  # prevent division by 0
        thetavec = invdist / np.sum(invdist)
        theta = squareform(thetavec)
        # multinomial sample
        n = np.shape(theta)[0]
        Z = np.zeros((n,n))
        samp = sampleLinks(q=thetavec, edgesToDraw=1, draws=draws)
        while np.sum(samp) < minEdges:
            draws = int(np.ceil(draws * 1.1))   #increase number of draws and try again
            samp = sampleLinks(q=thetavec,edgesToDraw=1,draws=draws)
        Z[np.tril_indices_from(Z, k =-1)] = samp
        
    elif sampleMethod == 'deterministic':
        theta = AATnn / AATnn.max()
        n = np.shape(AATnn)[0]
        sv = AATnn[np.tril_indices_from(AATnn, k =-1)]  #pull singular values from triangle
        cutOff = topNEdges(data = sv, minEdges = minEdges, 
                           n = n)['cutOff']
        Z = np.zeros((n,n))
        Z[np.where(AATnn >= cutOff)] = 1
        
    else:
        print('No valid sampleMethod selected. Please choose "Bernoulli", "multinomial", or "deterministic" .') 

    # NETWORK    
    # Create networkx graph from Z
    g = nx.Graph()
    
    #add nodes with colors of group
    for n in np.arange(np.shape(hm['corder'])[0]-1):
        g.add_node(hm['corder'][n],color=hm['group'][n])
    nodeColorList = list(nx.get_node_attributes(g,'color').values())
    
    #add edges with weight of theta (probability the link exists)
    cardE = len(np.where(Z==1)[1])
    edgeList = [(np.where(Z==1)[0][i], np.where(Z==1)[1][i]) for i in np.arange(cardE)]
    edgeWeightList = theta[np.where(Z==1)] * (2 / max(theta[np.where(Z==1)]))  #scaled link prob Pr(Z[i,j]=1) * weight
    for e in np.arange(len(edgeList)-1):
        g.add_edge(edgeList[e][0],edgeList[e][1],weight=edgeWeightList[e])

    # NODE SIZES
    # 1. cluster linkage importance
    #nodesizelist = cluster['linkage'] * (400 / max(cluster['linkage']))
    # 2. betweenness centrality (wide range of sizes; very small on periphery)
    #nodesizelist = np.asarray(list(nx.betweenness_centrality(G,normalized=False).values())) * (400 / max(list(nx.betweenness_centrality(G,normalized=False).values())))
    # 3. degree (smaller range of sizes; easier to see on the periphery)
    nodeSizeList = np.asarray(list(g.degree().values())) * (350 / max(list(g.degree().values())))   #scaled so the largest is size 350

    if plotting:
        # reproducibility
        np.random.seed(1)        
        
        #bc = nx.betweenness_centrality(g)
        E = len(nx.edges(g))
        V = len(g)
        k = round(E/V,3)
		
        #size = np.array(list(bc.values())) * 1000  
        # here replacing the hierarchical magnitude hm['corder']

        fignx = plt.figure(figsize=(10,10))
        ## use heatmap color groupings to color nodes and heatmap magnitudes to size nodes
        if layout == 'spring':
            nx.draw(g, pos=nx.spring_layout(g, scale=graphScale),
                    node_color=nodeColorList, node_size=nodeSizeList,
                    width=edgeWeightList)
        elif layout == 'fruchterman':
            nx.draw(g, pos=nx.fruchterman_reingold_layout(g, scale=graphScale),
                    node_color=nodeColorList, node_size=nodeSizeList,
                    width=edgeWeightList)
        else:
            print('Please indicate at a valid layout.')
        #else:
            #nx.graphviz_layout(g, prog=graphProg)
        plt.title('Network Created from Induced Rank = %s \n V = %s, E = %s, <k> = %s'%(r,V,E,k), fontweight='bold', fontsize=14)
    
        #plot log degree sequence
        degree_sequence=sorted(nx.degree(g).values(),reverse=True)
        fig3 = plt.figure(figsize=(10,5))
        plt.loglog(degree_sequence)
        plt.title('Log Degree Distribution', fontweight='bold', fontsize=14)
        
    return {'cluster':hm, 'graph':g, 'linkage':hm['linkage'], 'theta':theta, 'A':A, 'Z':Z}


def linkProbRank(x):
    """input: x matrix of link probabilities
    output: df with columns: 'i', 'j', 'prob' (probability i<->j)
    """
    #import numpy as np
    #import pandas as pd
    n = np.shape(x)[0]
    df = pd.DataFrame(np.zeros((n*(n-1)/2, 3)),columns=['i','j','prob'])
    index = 0
    for i in np.arange(n):
        for j in np.arange(n):
            if i > j:
                index += 1
                df.loc[index,'i'] = i
                df.loc[index,'j'] = j
                df.loc[index,'prob'] = x[i,j]
    df = df.sort('prob',axis=0,ascending = False)
    return df


def topNEdges(data, minEdges, n):
    """returns the sorted edges above cutOff such that number = minEdges
    """
    #import numpy as np
    Ep = minEdges / ( n*(n-1)/2 )  #minEdges proportion of total possible edges f(n)
    index = int(np.floor(len(data)*Ep))
    data = np.sort(data)
    data = data[::-1]  # decreasing order numpy array
    out = data[data > data[index]]
    return {'edgeList':out, 'cutOff':data[index]}


def buildTensor(df,r=1,offset=1):
    """ build an r-mode similarity adjacency tensor;
    Inputs: offset is number of columns in df to skip (shape(df)[1] =
    offset + r)
    df column  1 is ID;
    df columns [2:ncol] are the r relationship types;
    output: list of csr_matrix sparse matrices  
    (format for RESCAL_ALS input)
    """
    #import numpy as np
    #import pandas as pd
    #import sktensor as st

    n = np.shape(df)[0]
    X = []
    for k in np.arange(r):
        indices = indptr = data = []
        for i in np.arange(n):
            for j in np.arange(n):
                if i > j:
                    if df.iloc[i,offset+k] == df.iloc[j,offset+k]:  #skip offset col of mem_no
                        indices.append(i)
                        indptr.append(j)
                        data.append(1)  # always one if binary tensor
        # completed i,j loops; convert csr_matrix arguments to np.arrays
        indices = np.asarray(indices)
        indptr = np.asarray(indptr)
        data = np.asarray(data)
        X.append(st.csr_matrix( (data,(indices,indptr) ),
                               shape=(n,n),
                               dtype=int8) ) 
    return X