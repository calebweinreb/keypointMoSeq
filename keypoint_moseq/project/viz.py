import os
import cv2
import tqdm
import imageio
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 100

from vidio.read import OpenCVReader
from textwrap import fill

from keypoint_moseq.util import *
from keypoint_moseq.project.io import load_results


def get_edges(use_bodyparts, skeleton):
    edges = []
    for bp1,bp2 in skeleton:
        if bp1 in use_bodyparts and bp2 in use_bodyparts:
            edges.append([use_bodyparts.index(bp1),use_bodyparts.index(bp2)])
    return edges

def plot_scree(pca, savefig=True, project_dir=None):
    fig = plt.figure()
    plt.plot(np.arange(len(pca.mean_))+1,np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('PCs')
    plt.ylabel('Explained variance')
    plt.yticks(np.arange(0.5,1.01,.1))
    plt.xticks(range(0,len(pca.mean_)+2,2))
    plt.gcf().set_size_inches((2.5,2))
    plt.grid()
    plt.tight_layout()
    
    if savefig:
        assert project_dir is not None, fill(
            'The ``savefig`` option requires a ``project_dir``')
        plt.savefig(os.path.join(project_dir,'pca_scree.pdf'))
    plt.show()
          
def plot_pcs(pca, *, use_bodyparts, skeleton, keypoint_colormap,
             savefig=True, project_dir=None, scale=10, plot_n_pcs=10, 
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
        
        if savefig:
            assert project_dir is not None, fill(
                'The ``savefig`` option requires a ``project_dir``')
            plt.savefig(os.path.join(project_dir,f'pcs-{name}.pdf'))
        plt.show()
        

def plot_progress(model, data, history, iteration, path=None,
                  project_dir=None, name=None, savefig=True,
                  fig_size=None, seq_length=600, min_usage=.001, 
                  **kwargs):
    
    z = np.array(model['states']['z'])
    mask = np.array(data['mask'])
    durations = get_durations(z,mask)
    usages = get_usages(z,mask)
    
    history_iters = sorted(history.keys())
    past_stateseqs = [history[i]['states']['z'] 
                      for i in history_iters 
                      if 'states' in history[i]]
        
    if len(past_stateseqs)>0: 
        fig,axs = plt.subplots(1,4, gridspec_kw={'width_ratios':[1,1,1,3]})
        if fig_size is None: fig_size=(12,2.5)
    else: 
        fig,axs = plt.subplots(1,2)
        if fig_size is None: fig_size=(4,2.5)

    usages = np.sort(usages[usages>min_usage])[::-1]
    axs[0].bar(range(len(usages)),usages,width=1)
    axs[0].set_ylabel('probability')
    axs[0].set_xlabel('syllable rank')
    axs[0].set_title('Usage distribution')
    axs[0].set_yticks([])
    
    lim = int(np.percentile(durations, 95))
    binsize = max(int(np.floor(lim/30)),1)
    lim = lim-(lim%binsize)
    axs[1].hist(durations, range=(1,lim), bins=(int(lim/binsize)), density=True)
    axs[1].set_xlim([1,lim])
    axs[1].set_xlabel('syllable duration (frames)')
    axs[1].set_ylabel('probability')
    axs[1].set_title('Duration distribution')
    axs[1].set_yticks([])
    
    if len(past_stateseqs)>0:
        
        med_durs = [np.median(get_durations(z,mask)) for z in past_stateseqs]
        axs[2].scatter(history_iters,med_durs)
        axs[2].set_ylim([-1,np.max(med_durs)*1.1])
        axs[2].set_xlabel('iteration')
        axs[2].set_ylabel('duration')
        axs[2].set_title('median duration')
        
        nz = np.stack(np.array(mask[:,seq_length:]).nonzero(),axis=1)
        batch_ix,start = nz[np.random.randint(nz.shape[0])]
        seq_hist = np.stack([z[batch_ix,start:start+seq_length] for z in past_stateseqs])
        axs[3].imshow(seq_hist, cmap=plt.cm.jet, aspect='auto', interpolation='nearest')
        axs[3].set_xlabel('Time (frames)')
        axs[3].set_ylabel('Iterations')
        axs[3].set_title('Stateseq history')
        
    fig.suptitle(f'Iteration {iteration}')
    fig.set_size_inches(fig_size)
    plt.tight_layout()
    
    if savefig:
        if path is None:
            assert name is not None and project_dir is not None, fill(
                'The ``savefig`` option requires either a ``path`` '
                'or a ``name`` and ``project_dir``')
            path = os.path.join(project_dir,name,'fitting_progress.pdf')
        plt.savefig(path)  
    plt.show()
    
    
    



def crowd_movie_tile(key, start, end, videos, centroids, headings, 
                     dot_color=(255,255,255), window_size=112,
                     pre=30, post=60, dot_radius=4):
            
        cs = centroids[key][start-pre:start+post]
        h,c = headings[key][start],cs[pre]
        r = np.float32([[np.cos(h), np.sin(h)],[-np.sin(h), np.cos(h)]])
        c = r @ c - window_size//2
        M = [[ np.cos(h), np.sin(h),-c[0]], [-np.sin(h), np.cos(h),-c[1]]]
        
        tile = []
        frames = videos[key][start-pre:start+post]
        for ii,(frame,c) in enumerate(zip(frames,cs)):
            frame = cv2.warpAffine(frame,np.float32(M),(window_size,window_size))
            if 0 <= ii-pre <= end-start:
                pos = tuple([int(x) for x in M@np.append(c,1)])
                cv2.circle(frame, pos, dot_radius, dot_color, -1, cv2.LINE_AA)
            tile.append(frame)  
        return np.stack(tile)
    
    
def crowd_movie(instances, rows, cols, videos, centroids, headings,
                dot_color=(255,255,255), window_size=112, 
                pre=30, post=60, dot_radius=4):
    
    tiles = np.stack([
        crowd_movie_tile(
            key, start, end, videos, centroids, headings, 
            dot_color=dot_color, window_size=window_size,
            pre=pre, post=post, dot_radius=dot_radius
        ) for key, start, end in instances
    ]).reshape(rows, cols, post+pre, window_size, window_size, 3)
    return np.concatenate(np.concatenate(tiles,axis=2),axis=2)


    
def write_video_clip(frames, path, fps=30, quality=7):
            
    with imageio.get_writer(
        path, pixelformat='yuv420p', 
        fps=fps, quality=quality) as writer:

        for frame in frames: 
            writer.append_data(frame)


def generate_crowd_movies(
    results=None, output_dir=None, name=None, project_dir=None,
    results_path=None, video_dir=None, rows=4, cols=6, filter_size=9, 
    pre=30, post=60, min_usage=0.005, min_duration=3, dot_radius=4, 
    dot_color=(255,255,255), window_size=112, plot_keypoints=False, 
    use_reindexed=True, sampling_options={}, coordinates=None, 
    quality=7, **kwargs):
    
    assert video_dir is not None, fill(
        'The ``video_dir`` argument is required')
            
    if plot_keypoints:
        raise NotImplementedError()
        assert coordinates is not None, fill(
            '``coordinates`` are required when ``plot_keypoints==True``')        
    
    if output_dir is None:
        assert project_dir is not None and name is not None, fill(
            'Either specify the ``output_dir`` where crowd movies should '
            'be saved or include a ``project_dir`` and ``name``')
        output_dir = os.path.join(project_dir,name, 'crowd_movies')
    if not os.path.exists(output_dir): os.makedirs(output_dir)
    print(f'Writing crowd movies to {output_dir}')
    
    if results is None: results = load_results(
        name=name, project_dir=project_dir, path=results_path)
    
    video_paths = find_matching_videos(results.keys(), video_dir, as_dict=True)
    videos = {k: OpenCVReader(path) for k,path in video_paths.items()}
    fps = list(videos.values())[0].fps
    
    syllable_key = 'syllables' + ('_reindexed' if use_reindexed else '')
    syllables = {k:v[syllable_key] for k,v in results.items()}
    centroids = {k:median_filter(v['centroid'],(filter_size,1)) for k,v in results.items()}
    headings = {k:filter_angle(v['heading'], size=filter_size) for k,v in results.items()} 
    
    syllable_instances = get_syllable_instances(
        syllables, pre=pre, post=post, min_duration=min_duration)
    
    use_syllables = np.all([
        np.array(list(map(len,syllable_instances)))>=rows*cols, 
        get_usages(syllables)>=min_usage
    ], axis=0).nonzero()[0]
    
    for syllable in tqdm.tqdm(use_syllables, desc='Generating crowd movies'):
        
        instances = sample_syllable_instances(
            syllable_instances[syllable], rows*cols,
            coordinates=coordinates, **sampling_options)
        
        frames = crowd_movie(
            instances, rows, cols, videos, centroids, headings,
            dot_color=dot_color, window_size=window_size,
            pre=pre, post=post, dot_radius=dot_radius)

        path = os.path.join(output_dir, f'syllable{syllable}.mp4')
        write_video_clip(frames, path, fps=fps, quality=quality)

