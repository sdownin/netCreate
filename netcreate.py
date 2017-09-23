# -*- coding: utf-8 -*-
"""

netcreate

A module to compute network tie strengths from attribute similarities of members. 
Network sampling and plotting methods are included.

Copyright (C) 2015 Stephen Downing <sdowning.bm02g@nctu.edu.tw>
This program is provided without any warranty.
"""

import numpy as np
import matplotlib.pyplot as plt
import logging
import pandas as pd
import networkx as nx
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix
from time import time
from sktensor.rescal import als  # include Max Nickel's rescal.py in working directory
from cnames import cnames # color name, html code dictionary

#-------------- general functions --------------------------------------------

def heatmap(dm, plotting=True, color_threshold=0.7, metric='euclidean', linkage_method='complete', *args, **kwargs):
    """ Creates a heatmap from hierarchical agglomertatic clustering of data matrix
    
    Parameters
    ----------
    dm : matrix
        data matrix   
    plotting : logical 
        output heatmap plot
    color_threshold : float [0,1]
        cluster distance custoff; percent of largest distance (larger --> fewer clusters; smaller --> more cluster)
    metric : string
        distance metric to use (eg, 'euclidean' is L2-norm)
    linkage_method : string
        hierarchical clustering linkage method (eg, 'complete' is max dist between each cluster)
    
    Returns
    ----------
    dict
        distmat : ndarray
            pointwise distance matrix
        group : list
            group membership (color abbreviation)
        linkage : ndarray
            hierarchical clustering endocded as matrix with columns: [member_i, member_j, dist_ij, cluster]
        rorder : list
            leaves of dengrogram (right oriented)
        corder : list
            leaves of dengrogram (center oriented)
    """
    np.random.seed(111)
    np.random.shuffle(cnames)    
    
    D1 = squareform(pdist(dm, metric=metric))
    D2 = squareform(pdist(dm.T, metric=metric))
    
    if plotting:
        f = plt.figure(figsize=(6, 6))

    # add first dendrogram
    if plotting:
        ax1 = f.add_axes([0.09, 0.1, 0.2, 0.6])
        ax1.set_xticks([])
        ax1.set_yticks([])
    Y = linkage(D1, method=linkage_method) 
    Z1 = dendrogram(Y, orientation='right', color_threshold=color_threshold*max(Y[:,2]),
                    link_color_func=lambda k: cnames[k])

    # add second dendrogram
    if plotting:
        ax2 = f.add_axes([0.3, 0.71, 0.6, 0.2])
        ax2.set_xticks([])
        ax2.set_yticks([])
    Y = linkage(D2, method='complete')
    Z2 = dendrogram(Y, color_threshold=color_threshold*max(Y[:,2]),
                    link_color_func=lambda k: cnames[k])

    # add matrix plot
    idx1 = Z1['leaves']
    idx2 = Z2['leaves']
    D = D1[idx1, :]
    D = D[:, idx2]
    if plotting:
        axmatrix = f.add_axes([0.3, 0.1, 0.6, 0.6])
        im = axmatrix.matshow(D, aspect='auto', origin='lower', cmap='hot')
        axmatrix.set_xticks([])
        axmatrix.set_yticks([])
    
    return {'distmat' : D, 'rorder' : Z1['leaves'], 'corder' : Z2['leaves'], 
            'group':Z1['color_list'], 'linkage':Y}

def top_n_edges(data, minEdges, n, *args, **kwargs):
    """ returns the sorted edges above cutOff such that number = minEdges
    
    Parameters
    ----------
    data : ndarray (or list)
        vector of pairwise distances
    minEdges : int
        number of edges (social ties) to be assigned in the network
    n : int
        number of members in the network
        
    Returns
    ----------
    dict
        edgelist : list
            sorted pairwise distances representing strength of edges (ties) between network members
        cutoff : int
            distance cutoff above which ties are assigned to equal minEdges in network
    """
    #import numpy as np
    Ep = minEdges / ( n*(n-1)/2 )  #minEdges proportion of total possible edges f(n)
    index = int(np.floor(len(data)*Ep))
    data = np.sort(data)
    data = data[::-1]  # decreasing order numpy array
    out = data[data > data[index]]
    return {'edgeList':out, 'cutOff':data[index]}
   
def net_sample_Bernoulli(AATnn, minEdges, *args, **kwargs):
    """ NETWORK SAMPLE ALGORITHM:
    random sample ties in network adjacency matrix
    one-element-at-a-time Bernoulli shortcut
    instead of multinomial sample of entire adjacency
    
    Parameters
    ----------
    AATnn : ndarray
        outer product matrix (ie, covariance) of eigenvectors from RESCAL_ALS tensor decomposition with negative values replaced by zeros
    minEdges : int
        number of edges (social ties) to be assigned in the network
        
    Returns
    ----------
    tuple
        tie probabilities : ndarray
            pairwise distances normalized by largest distance
        sampled network : ndarray
            binary matrix of assigned ties above cutoff yielding at least minEdges
    """
    # shortcut: normalize by largest AAT (non-negative) value for separate bernoulli draws
    # instead of summing over all AAT for multinomial draw, which is harder to 
    # transfer back and forth between vector and triangular matrix
    theta = AATnn / AATnn.max()

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
                if np.sum(Z) > minEdges:
                    break
            if np.sum(Z) > minEdges:
                break
    return (theta, Z)

def net_sample_multinomial(A, minEdges, edgesPerSample=1, *args, **kwargs):
    """ NETWORK SAMPLING ALGORITHM:
    sample networks ties from multinomial distribution
    defined as 1/AAT[i,j] normalized by  sum(AAT[i>j])
    PROBLEM: doesn't sufficiently cluster the resulting network doesn't return exact number of ties, only at least as many as specified minEdges
    
    Parameters
    ----------
    A : ndarray
        matrix of eigenvectors from RESCAL_ALS tensor decomposition with negative values replaced by zeros
    minEdges : int
        number of edges (social ties) to be assigned in the network
        
    Returns
    ----------
    tuple
        tie probabilities : ndarray
            pairwise distances normalized by largest distance
        sampled network : ndarray
            binary matrix of assigned ties above cutoff yielding at least minEdges    
    """
    draws = int(np.ceil(minEdges*1.2))
    # pairwise distances between observations
    dist = pdist(A)   # what matrix to use:  pdist(A) or just tril(AAT) directly?
    invdist = dist
    invdist[invdist != 0] = 1/invdist[invdist!=0]  # prevent division by 0
    thetavec = invdist / np.sum(invdist)
    theta = squareform(thetavec)
    
    # multinomial sample
    n = np.shape(theta)[0]
    Z = np.zeros((n,n))
    # samp = sampleLinks(q=thetavec, edgesToDraw=1, draws=draws)
    y = np.random.multinomial(edgesPerSample, thetavec, draws)
    samp = np.asarray([np.mean([y[draw][item] for draw in range(draws)]) for item in range(len(thetavec))])
    samp = np.ceil(samp)
    
    # repeat until reaching enough network ties
    while np.sum(samp) < minEdges:
        draws = int(np.ceil(draws * 1.1))   #increase number of draws and try again
        #samp = sampleLinks(q=thetavec,edgesToDraw=1,draws=draws)
        y = np.random.multinomial(edgesPerSample, thetavec, draws)
        samp = np.asarray([np.mean([y[draw][item] for draw in range(draws)]) for item in range(len(thetavec))])
        samp = np.ceil(samp)
    
    Z[np.tril_indices_from(Z, k =-1)] = samp
    
    return (theta, Z)
    
def net_sample_deterministic(AATnn, minEdges, *args, **kwargs):
    """ NETWORK SAMPLING ALGORITHM:
    Assigns network ties to top [minEdges] relationships between members
    
   Parameters
    ----------
    AATnn : ndarray
        outer product matrix (ie, covariance) of eigenvectors from RESCAL_ALS tensor decomposition with negative values replaced by zeros
    minEdges : int
        number of edges (social ties) to be assigned in the network
        
    Returns
    ----------
    tuple
        tie probabilities : ndarray
            pairwise distances normalized by largest distance
        sampled network : ndarray
            binary matrix of assigned ties above cutoff yielding at least minEdges
    """
    theta = AATnn / AATnn.max()
    n = np.shape(AATnn)[0]
    sv = AATnn[np.tril_indices_from(AATnn, k =-1)]  #pull singular values from triangle
    cutOff = top_n_edges(data = sv, minEdges = minEdges, 
                       n = n)['cutOff']
    Z = np.zeros((n,n))
    Z[np.where(AATnn >= cutOff)] = 1
    
    return (theta, Z)

def link_prob_rank(x, *args, **kwargs):
    """ Creates a ranking of pairwise relationship strengths
    
    Parameters
    ----------
    x : ndarray
        matrix of link probabilities
   
    Returns
    ----------
    ranking : ndarray
        df with columns: [i, j, probability i<->j]
    """
    #import numpy as np
    #import pandas as pd
    n = np.shape(x)[0]
    df = pd.DataFrame(np.zeros((n*(n-1)/2, 3)),columns=['i','j','prob'])
    index = 0
    for i in range(n):
        for j in range(n):
            if i > j:
                index += 1
                df.loc[index,'i'] = i
                df.loc[index,'j'] = j
                df.loc[index,'prob'] = x[i,j]
    df = df.sort('prob',axis=0,ascending = False)
    return df  

def net_plot(title, AAT, theta, Z, r, lambda_A, lambda_R, layout='fruchterman', plotting = True, graphScale=1.0, color_threshold=0.7, *args, **kwargs):
    """ Plots a sample from a network created using netCreate method.
    Provide the eigenvector covariances AAT from RESCAL_ALS output and Z the sampled network from one of the netCreate sampling algorithms

    Parameters
    ----------
    title : str
        title of figure to save plot
    AAT : ndarray
        outer product matrix (ie, covariance) of eigenvectors from RESCAL_ALS tensor decomposition
    theta : ndarray
        pairwise distances normalized by largest distance 
    Z : ndarray
        binary matrix of assigned ties above cutoff yielding at least minEdges
    r : int
        induced rank of similarity tensor decomposition
    lambda_A : float
        tensor decomposition regularization paramter for matrix A
    lambda_R : float
        tensor decomposition regularization paramter for reduced tensor R
    layout : str
        graph layout to plot
    plotting : logical
        output graph plot
    graph_scale : float
        scale factor for the size of graph
    color_threshold : float [0,1]
        inter-cluster cutoff as percentage of largest cluster distance
        
    Returns
    ----------
    dict
        distmat : ndarray
             pointwise distance matrix
        graph : networkx graph
            network sample graph object
        group : list
            group membership (color abbreviation)
        linkage : ndarray
            hierarchical clustering endocded as matrix with columns: [member_i, member_j, dist_ij, cluster]  
    """        
    # get system time to name figures        
    time0 = str(int(time())  )
    time0 = time0[(len(time0)-5):(len(time0))]

    # heatmap
    hm = heatmap(AAT, plotting=plotting, color_threshold=color_threshold)
    if plotting:
        plt.suptitle(r'A(A^T) HAC for Induced Rank = %s, $\lambda_{A}$ = %s, $\lambda_{R}$ = %s ' %(r,lambda_A, lambda_R), fontweight='bold', fontsize=14)
        plt.savefig(title+'_heatmap_'+time0, figsize=(6,6))
    
    # NETWORK    
    # Create networkx graph from Z
    g = nx.Graph()
   
    #add nodes with colors of group
    for n in range(np.shape(hm['corder'])[0]-1):
        g.add_node(hm['corder'][n],color=hm['group'][n])
    nodeColorList = list(nx.get_node_attributes(g,'color').values())
    
    #add edges with weight of theta (probability the link exists)
    cardE = len(np.where(Z==1)[1])
    edgeList = [(np.where(Z==1)[0][i], np.where(Z==1)[1][i]) for i in range(cardE)]
    edgeWeightList = theta[np.where(Z==1)] * (2 / max(theta[np.where(Z==1)]))  #scaled link prob Pr(Z[i,j]=1) * weight
    for e in range(len(edgeList)-1):
        g.add_edge(edgeList[e][0],edgeList[e][1],weight=edgeWeightList[e])

    ## NODE SIZE (uncomment one to use)
    ## 1. cluster linkage importance
    #nodesizelist = cluster['linkage'] * (400 / max(cluster['linkage']))
    ## 2. betweenness centrality (wide range of sizes; very small on periphery)
    #nodesizelist = np.asarray(list(nx.betweenness_centrality(G,normalized=False).values())) * (400 / max(list(nx.betweenness_centrality(G,normalized=False).values())))
    # 3. degree (smaller range of sizes; easier to see on the periphery)
    nodeSizeList = np.asarray(list(g.degree().values())) * (300 / max(list(g.degree().values())))   #scaled so the largest is size 350

    # reproducibility
    np.random.seed(1)        
    
    #bc = nx.betweenness_centrality(g)
    E = len(nx.edges(g))
    V = len(g)
    k = round(E/V,3)
		
    #size = np.array(list(bc.values())) * 1000  
    # here replacing the hierarchical magnitude hm['corder']

    fignx = plt.figure(figsize=(6,6))
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
    plt.savefig(title+'_graph_'+time0, figsize=(6,6))

    ## uncomment to plot log degree sequence
    #degree_sequence=sorted(nx.degree(g).values(),reverse=True)
    #fig3 = plt.figure(figsize=(5,3))
    #plt.loglog(degree_sequence)
    #plt.title('Log Degree Distribution', fontweight='bold', fontsize=14)
    
    return {'distmat':hm['distmat'], 'graph':g, 'linkage':hm['linkage'], 
            'group':hm['group'], 'Y': Y}
  
#----------------------- netCreate class -------------------------------------

class netCreate:
    """A class for objects created from pairwise attribute similarities,
    with functions to predict network from similarity adjacency tensor 
    using RESCAL_ALS
    
    Examples
    ----------
    >>> # 100-person network of 4 groups using 10 attributes to build similarity tensor
    >>> np.random.seed(1)  # for reproducibility
    >>> df1 = np.random.choice((0,1,2),size=25*10,replace=True).reshape((25,10))
    >>> df2 = np.random.choice((2,3),size=25*10,replace=True).reshape((25,10))
    >>> df3 = np.random.choice((3,4),size=25*10,replace=True).reshape((25,10))
    >>> df4 = np.random.choice((4,5,6),size=25*10,replace=True).reshape((25,10))
    >>> df = np.vstack((df1,df2,df3,df4))
    >>> # mix up the groups within the dataframe
    >>> index = np.random.choice(np.arange(100), size=100,replace=False)
    >>> df = df[index,:]  # mix up the groups within the dataframe
    >>> # create netCreate object instance of netCreate class
    >>> a = netCreate()
    >>> # STEP 1: build similiarity tensor 
    >>> a.build_sim_tensor(x=df)
    >>> # STEP 2: decompose similarity tensor for interpersonal distances in reduced space
    >>> a.decompose_tensor(rank=df.shape[0]-3, lambda_A=6, lambda_R=6)
    >>> # STEP 3: create network  (avg. degree of 5 relationships per person)
    >>> a.net_create(minEdges=df.shape[0]*5, color_threshold=0.7,Bernoulli=True) 
    """
    def __init__(self, X = None):  
        if X is not None:
            self.X = X
        else: 
            self.X = None
            
        self.network = {}
        self.graph = {}
        self.pred_rank = {}
        self.theta = {}
        self.cluster = {}
        
    def __repr__(self):
        return "netCreate()"
        
    def __str__(self):
        return "<From str method of netCreate:\nX: %s \nnetwork: %s \ngraph: %s \npred_rank: %s\n>" % (type(self.X),self.network.keys(), self.graph.keys(), self.pred_rank.keys())
        
    def build_sim_tensor(self, x, offset=0, matchValue=1, Sim=True, metric='hamming', *args, **kwargs):
        """ STEP 1 : build
            build an r-mode similarity adjacency tensor
        
        Parameters
        ----------
        x : ndarray
            input dataframe
        offset : int
            number of columns in x to skip (shape(df)[1] = offset + r) where r = number of r relationship types
        matchValue : int or float
            values of similarity matrix elements for which matches between members represent pairwise similarities
        Sim : logical
            using similarity matrix implies 1 - distances (else distance matrix)
        metric : str
            distance metric to use (eg, 'hamming' is proportion of dissimilar elements in vector, which for length-1 vector returns either 0 or 1 )
        
        Returns
        ----------
        X : list
            list of csr_matrix sparse matrices (format for RESCAL_ALS input); matrices in list are tensor frontal slices X_k for k in 1...r)
        """
        time0 = time()
        X = []
        discarded = []
        for j in range(offset,x.shape[1]):
            if isinstance(x, (pd.DataFrame, pd.Series)):  # checking if Pandas object
                if Sim:
                    simmat = 1 - squareform(pdist(np.asarray(x.iloc[:,(j-1):j]), metric) )
                else:
                    simmat = squareform(pdist(np.asarray(x.iloc[:,(j-1):j]), metric) )
            else:
                if Sim:
                    simmat = 1 - squareform(pdist(np.asarray(x[:,(j-1):j]), metric) )
                else:
                    simmat = squareform(pdist(np.asarray(x[:,(j-1):j]), metric) )
            simmat[np.triu_indices(simmat.shape[0],k=0)] = None
            indices = np.where(simmat==matchValue)
            n = x.shape[0]
            holder = csr_matrix( ( np.ones(indices[0].shape[0]), indices ),
                                      shape=(n,n), dtype=np.int8)
                                      
            if holder.getnnz() > 0:
                X.append(holder)
            else:
                discarded.append(j)
            # update progress
            if j % np.ceil((x.shape[1]-offset)/10) == 0 :
                    print('completed feature %s' % ( j ) ) 
        
        print('The following feature indices were discarded containing 0 similarities:')
        print(discarded)        
        
        timeout = time() - time0   
        print('\nTensor build elapsed time: %s seconds' % ( round(timeout,3) ))    
        self.X = X
    
    def decompose_tensor(self, rank, X=None, init='nvecs', lambda_A=10, lambda_R=10, *args, **kwargs):
        """ STEP 2 : decompose
            Decompose adjacency tensor using RESCAL_ALS (see RESCAL documentation)
        
        Parameters
        ----------
        rank : int
            induced rank of decomposed similarity tensor
        X : list
            list of matrices X_k, for k in 1...r (where r is number of relationships), which are frontal slices of similarity tensor
        init : str
            RESCAL_ALS parameter for initializing tensor decomposition values
        lambda_A : float
            tensor decomposition regularization paramter for matrix A
        lambda_R : float
            tensor decomposition regularization paramter for reduced tensor R
            
        Returns
        ----------
        A : ndarray
            similarity tensor decomposition eigenvectors (from RESCAL_ALS)
        R : ndarray
            similarity tensor decomposition reduced tensor (from RESCAL_ALS)
        AAT : ndarray
            outer product of A matrix (AATnn replaces negative values with zeros)
        SIF : ndarray
            Social Influence Factor matrix (strength of pairwise social relations)
        
        """   
        # if X not specified, used X attribute if exists
        if X is None:
            if hasattr(self, 'X'):
                X = self.X
            else:
                print("netCreate object has no X attribute. Need to Provide X argument.")
            
        # Set logging to INFO to see RESCAL information
        logging.basicConfig(level=logging.INFO)
        A, R, fit, itr, exectimes = als(X, rank=rank, init=init, lambda_A=lambda_A, lambda_R=lambda_R)
        self.rescal_params = {'rank':rank,'fit':fit,'lambda_A':lambda_A,'lambda_R':lambda_R}        
        self.A = A
        self.R = R   
        self.AAT = AATnn = np.dot(A,A.T)
        AATnn[AATnn < 0] = 0
        
        #make SIF: zeros on diags, non-negative values both upper & lower triangles
        # not row-normalized since some individuals are more likely to have more ties 
        SIF = AATnn 
        SIF[np.diag_indices(SIF.shape[0])] = 0
        self.SIF = SIF
        
        # remove upper triangle by keeping only lower triangle indexed values
        self.AATnn = np.tril(AATnn, k= -1)  # k = -1 to keep only below diagonal    
        
    def net_create(self, minEdges, X=None, deterministic=False, Bernoulli=False, multinomial=False, plotting=True, layout='spring', graphScale=1.0, color_threshold=0.7, *args, **kwargs): 
        """Wrapper function for RESCAL_ALS. Create network by given sampleMethod from hierarchical clustering of RESCALA_ALS 
        tensor factorization of singular values matrix A(A^T)
        
        Parameters
        ----------
        X = list
            list of matrices X_k, for k in 1...r (where r is number of relationships), which are frontal slices of similarity tensor
        minEdges : int
            number of edges (social ties) to be assigned in the network
        deterministic : logical
            compute network sample using deterministic sampling method
        Bernoulli : logical
            compute network sample using Bernoulli sampling method
        multinomial : logical
            compute network sample using multinomial sampling method
        plotting : logical
            output graph plot
        layout : str
            graph layout to plot
        graph_scale : float
            scale factor for the size of graph
        color_threshold : float [0,1]
            inter-cluster cutoff as percentage of largest cluster distance
        
        Returns
        ----------
        pred_rank : ndarray (see link_prob_rank())
            ranking of pairwise relationship strengths in dataframe with columns: [i, j, probability i<->j] (for each sampling method set to True)
        theta : ndarray
            tie probabilities computed as pairwise distances normalized by largest distance (for each sampling method set to True)
        graph : networkx graph object
            sampled graph (for each sampling method set to True)
        cluster : 
            cluster membership (for each sampling method set to True)
        """

        if not any([deterministic, Bernoulli, multinomial]):
            print("Select at least one sampling method.")
            return
        
        # warn and end if not yet decomposed tensor
        if not hasattr(self,'AAT'):
            print("AAT matrix of eigenvector covariance not defined. First decompose tensor; then create network")
            return
        # if X not specified, used X attribute if exists
        if X is None:
            if hasattr(self, 'X'):
                X = self.X
            else:
                print("netCreate object has no X attribute. Need to Provide X argument.")
        
        # sample networks by given sampling methods
        if Bernoulli:
            np.random.seed(1)
            theta,  self.network['Bernoulli'] = net_sample_Bernoulli(self.AATnn, minEdges=minEdges, *args, **kwargs)
            self.pred_rank['Bernoulli'] = link_prob_rank(theta)            
            self.theta['Bernoulli'] = theta            
            if plotting:
                plotted_Bernoulli = net_plot('Bern',self.AAT, theta, self.network['Bernoulli'], self.rescal_params['rank'], self.rescal_params['lambda_A'], self.rescal_params['lambda_R'], layout=layout, color_threshold=color_threshold, *args, **kwargs)              
                self.graph['Bernoulli'] = plotted_Bernoulli['graph']
                self.cluster['Bernoulli'] = plotted_Bernoulli['group']
                
        if deterministic:
            np.random.seed(1)
            theta,  self.network['deterministic'] = net_sample_deterministic(self.AATnn, minEdges=minEdges, *args, **kwargs)     
            self.pred_rank['deterministic'] = link_prob_rank(theta)            
            self.theta['deterministic'] = theta             
            if plotting:
                plotted_deterministic = net_plot('determ',self.AAT, theta, self.network['deterministic'], self.rescal_params['rank'], self.rescal_params['lambda_A'], self.rescal_params['lambda_R'], layout=layout, color_threshold=color_threshold, *args, **kwargs)              
                self.graph['deterministic'] = plotted_deterministic['graph']
                self.cluster['deterministic'] = plotted_deterministic['group']                 
                  
        if multinomial:
            np.random.seed(1)
            theta,  self.network['multinomial'] = net_sample_multinomial(self.AATnn, minEdges=minEdges, *args, **kwargs)           
            self.pred_rank['multinomial'] = link_prob_rank(theta)            
            self.theta['multinomial'] = theta            
            if plotting:
                plotted_multinomial = net_plot('multi',self.AAT, theta, self.network['multinomial'], self.rescal_params['rank'], self.rescal_params['lambda_A'], self.rescal_params['lambda_R'], layout=layout, color_threshold=color_threshold, *args, **kwargs)    
                self.graph['multinomial'] = plotted_multinomial['graph']
                self.cluster['multinomial'] = plotted_multinomial['group']
    





#--------------------------------------------------------------------
#  previous version of build_sim_tensor() method
#--------------------------------------------------------------------
#    def build_sim_tensor(self, df, offset=1, *args, **kwargs):
#        """ build an r-mode similarity adjacency tensor;
#        Inputs: offset is number of columns in df to skip (shape(df)[1] =
#        offset + r)
#        df column  1 is ID;
#        df columns [offset:ncol] are the r relationship types;
#        output: list of csr_matrix sparse matrices  
#        (format for RESCAL_ALS input)
#        """
#        #import numpy as np
#        #import pandas as pd
#        #import sktensor as st
#        n = np.shape(df)[0]
#        r = df.shape[1] - offset  #exluding offset columns; use r remaining columns
#        X = []
#        # loop over r features
#        for k in range(r):
#            indices = []
#            indptr = []
#            data = []
#            # loop through each pair of members only once
#            for i in np.arange(n):
#                for j in np.arange(n):
#                    xi = df.iloc[i,offset+k]
#                    xj = df.iloc[j,offset+k]
#                    if i > j and None not in [xi,xj] and 'NA' not in [xi,xj]:  #lower triangle matrix
#                        # check data type
#                        # STRING
#                        if isinstance(df.iloc[i,offset+k], str):
#                            if df.iloc[i,offset+k] == df.iloc[j,offset+k]:  #skip offset col of mem_no
#                                indices.append(i)
#                                indptr.append(j)
#                                data.append(1)  # always one if binary tensor
#                        # NUMERIC: [FLOAT, INT]
#                        # elif isinstance(df.iloc[i,offset+k], float) or isinstance(df.iloc[i,offset+k], int):
##                        else:
##                            sigma = np.std(~np.isnan(df.iloc[:,offset+k]))
##                            a = np.linalg.norm(df.iloc[i,offset+k])
##                            b = np.linalg.norm(df.iloc[j,offset+k])
##                            # count as similar if within 1 std.dev.
##                            if np.abs(a - b) <= 4*sigma:
##                                indices.append(i)
##                                indptr.append(j)
##                                data.append(1)  # always one if binary tensor
#                if i % 10 == 0:
#                    print('k: %s, i: %s, j: %s'%(k,i,j))                          
#                        
#            # completed i,j loops; convert csr_matrix arguments to np.arrays
#            indices = np.asarray(indices)
#            indptr = np.asarray(indptr)
#            data = np.asarray(data)
#            X.append(st.csr_matrix( (data,(indices,indptr) ),
#                                   shape=(n,n),
#                                   dtype=np.int8) ) 
#            # update progress at most 10 times
#            if k % np.ceil(r/20) == 0:
#                print('\ncompleted feature %s out of %s'%(k+1,r))
#                
#        self.X = X 
#        return {'row':indices, 'col':indptr, 'data':data} 