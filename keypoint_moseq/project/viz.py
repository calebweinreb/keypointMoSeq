import numpy as np
import os
from textwrap import wrap
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 100
from keypoint_moseq.util import center_embedding, get_durations, get_usages

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
          
def plot_pcs(pca, *, use_bodyparts, skeleton, keypoint_colormap,
             project_directory=None, scale=10, plot_n_pcs=10, 
             axis_size=(2,1.5), ncols=5, node_size=20, **kwargs):
    
    k = len(use_bodyparts)
    d = len(pca.mean_)//(k-1)  
    Gamma = np.array(center_embedding(k))
    edges = get_edges(use_bodyparts, skeleton)
    cmap = plt.cm.get_cmap(keypoint_colormap)
    plot_n_pcs = min(plot_n_pcs, pca.components_.shape[0])
    
    if d==2: dims_list,names = [[0,1]],['xy']
    if d==3: dims_list,names = [[0,1],[1,2]],['xy','yz']
    
    for dims,name in zip(dims_list,names):
        nrows = int(np.ceil(plot_n_pcs/ncols))
        fig,axs = plt.subplots(nrows, ncols, sharex=True, sharey=True)
        for i,ax in enumerate(axs.flat):
            ymean = Gamma @ pca.mean_.reshape(k-1,d)[:,dims]
            y = Gamma @ (pca.mean_ + scale*pca.components_[i]).reshape(k-1,d)[:,dims]
            for e in edges: ax.plot(*ymean[e].T, color=cmap(e[0]/(k-1)), zorder=0, alpha=0.25)
            ax.scatter(*ymean.T, c=np.arange(k), cmap=cmap, s=node_size, zorder=1, alpha=0.25, linewidth=0)
            for e in edges: ax.plot(*y[e].T, color=cmap(e[0]/(k-1)), zorder=2)
            ax.scatter(*y.T, c=np.arange(k), cmap=cmap, s=node_size, zorder=3)
            ax.set_title(f'PC {i+1}', fontsize=10)
            ax.set_aspect('equal')
            ax.axis('off')
        
        fig.set_size_inches((axis_size[0]*ncols, axis_size[1]*nrows))
        plt.tight_layout()
        
        if project_directory is not None:
            plt.savefig(os.path.join(project_directory,'figures',f'pcs-{name}.pdf'))
        

def plot_progress(*, history, mask, states, iteration, 
                  project_directory=None, name=None, save=False,
                  fig_size=None, seq_length=600, **kwargs):
    
    z,mask = np.array(states['z']),np.array(mask)
    durations = get_durations(z,mask)
    usages = get_usages(z,mask)
    
    if len(history['z'])>0: 
        fig,axs = plt.subplots(1,4, gridspec_kw={'width_ratios':[1,1,1,3]})
        if fig_size is None: fig_size=(12,2)
    elif len(history['median_duration'])>0: 
        fig,axs = plt.subplots(1,3)
        if fig_size is None: fig_size=(6,2)
    else: 
        fig,axs = plt.subplots(1,2)
        if fig_size is None: fig_size=(4,2)

    axs[0].bar(range(len(usages)),sorted(usages, reverse=True))
    axs[0].set_ylabel('probability')
    axs[0].set_xlabel('syllable rank')
    axs[0].set_title('Usage distribution')
    axs[0].set_yticks([])
    
    lim = np.percentile(durations, 95)
    axs[1].hist(durations, range=(0,lim), bins=int(min(lim,30)), density=True)
    axs[1].set_xlabel('syllable duration (frames)')
    axs[1].set_ylabel('probability')
    axs[1].set_title('Duration distribution')
    axs[1].set_yticks([])
    
    if len(history['median_duration'])>0:
        axs[2].scatter(history['iteration'],history['median_duration'])
        axs[2].set_ylim([-1,np.max(history['median_duration'])*1.1])
        axs[2].set_xlabel('iteration')
        axs[2].set_ylabel('duration')
        axs[2].set_title('median duration')
        
    if len(history['z'])>0:
        nz = np.stack(np.array(mask[:,seq_length:]).nonzero(),axis=1)
        batch_ix,start = nz[np.random.randint(nz.shape[0])]
        seq_hist = np.array(history['z'])[:,batch_ix,start:start+seq_length]
        axs[3].imshow(seq_hist, cmap=plt.cm.jet, aspect='auto', interpolation='nearest')
        axs[3].set_xlabel('Time (frames)')
        axs[3].set_ylabel('Iterations')
        axs[3].set_title('Stateseq history')

    fig.set_size_inches(fig_size)
    plt.tight_layout()
    if save:
        assert name and project_directory, wrap(
            'Cannot save figure if ``name`` or ``project_directory`` '
            'is None. Provide these arguments or set ``save=False``')
        save_path = os.path.join(project_directory,'figures',name)+'.pdf'
        plt.savefig(save_path, dpi=300)  
    plt.show()
    
