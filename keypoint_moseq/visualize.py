import numpy as np
import os
import matplotlib.pyplot as plt
from keypoint_moseq.util import center_embedding

def get_edges(use_bodyparts, skeleton):
    edges = []
    for bp1,bp2 in skeleton:
        if bp1 in use_bodyparts and bp2 in use_bodyparts:
            edges.append([use_bodyparts.index(bp1),use_bodyparts.index(bp2)])
    return edges

def plot_scree(pca, project_directory=None):
    fig = plt.figure()
    plt.plot(np.arange(len(pca.mean_))+1,np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('PCs')
    plt.ylabel('Explained variance')
    plt.yticks(np.arange(0.5,1.01,.1))
    plt.xticks(range(0,len(pca.mean_)+2,2))
    plt.gcf().set_size_inches((2.5,2))
    plt.grid()
    plt.tight_layout()
    if project_directory is not None:
        plt.savefig(os.path.join(project_directory,'figures','pca_scree.pdf'))
          
def plot_pcs(pca, project_directory=None, scale=10, plot_n_pcs=10, *,
             use_bodyparts, skeleton, keypoint_colormap, **config):
    
    k = len(use_bodyparts)
    d = len(pca.mean_)//(k-1)  
    Gamma = np.array(center_embedding(k))
    edges = get_edges(use_bodyparts, skeleton)
    cmap = plt.cm.get_cmap(keypoint_colormap)
    plot_n_pcs = min(plot_n_pcs, pca.components_.shape[0])
    
    if d==2: dims_list,names = [[0,1]],['xy']
    if d==3: dims_list,names = [[0,1],[1,2]],['xy','yz']
    
    for dims,name in zip(dims_list,names):
        fig,axs = plt.subplots(2,int(np.ceil(plot_n_pcs/2)), sharex=True, sharey=True)
        for i,ax in enumerate(axs.flat):
            ymean = Gamma @ pca.mean_.reshape(k-1,d)[:,dims]
            y = Gamma @ (pca.mean_ + scale*pca.components_[i]).reshape(k-1,d)[:,dims]
            for e in edges: ax.plot(*ymean[e].T, color=cmap(e[0]/(k-1)), zorder=0, alpha=0.2)
            ax.scatter(*ymean.T, c=np.arange(k), cmap=cmap, s=50, zorder=1, alpha=0.2, linewidth=0)
            for e in edges: ax.plot(*y[e].T, color=cmap(e[0]/(k-1)), zorder=2)
            ax.scatter(*y.T, c=np.arange(k), cmap=cmap, s=50, zorder=3)
            ax.set_aspect('equal')
            ax.axis('off')
        plt.tight_layout()
        fig.set_size_inches((plot_n_pcs,2))
        if project_directory is not None:
            plt.savefig(os.path.join(project_directory,'figures',f'pcs-{name}.pdf'))
