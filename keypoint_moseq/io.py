from jax.tree_util import tree_map
import jax.numpy as jnp
import numpy as np
import joblib
import yaml
import os
from textwrap import fill

def build_yaml(sections, comments):
    text_blocks = []
    for title,data in sections:
        centered_title = f' {title} '.center(50, '=')
        text_blocks.append(f"\n\n{'#'}{centered_title}{'#'}")
        for key,value in data.items():
            text = yaml.dump({key:value}).strip('\n')
            if key in comments: text = f"\n{'#'} {comments[key]}\n{text}"
            text_blocks.append(text)
    return '\n'.join(text_blocks)
        

def generate_config(project_directory, **kwargs):
    """
    Generate a config.yml file with project settings.
    Default settings will be used unless overriden by 
    a keywork argument.
    
    Parameters
    ----------
    project_directory: str 
        A file ``config.yml`` will be generated in this directory.
    
    **kwargs
        Custom project settings.
        
    """
    
    def update_dict(new, original):
        return {k:new[k] if k in new else v for k,v in original.items()} 
    
    hypperams = {k: update_dict(kwargs,v) for k,v in {
        'error_estimator': {'A':1, 'K':100, 'B':-20, 'M':0.4},
        'obs_hypparams': {'sigmasq_0':10, 'sigmasq_C':.1, 'nu_sigma':1e5, 'nu_s':5},
        'ar_hypparams': {'nlags': 3, 'S_0_scale': 0.01, 'K_0_scale': 10.0},
        'trans_hypparams': {'num_states': 100, 'gamma': 1e3, 'alpha': 5.7, 'kappa': 1e6},
        'cen_hypparams': {'sigmasq_loc': 0.5}
    }.items()}

    anatomy = update_dict(kwargs, {
        'bodyparts': ['BODYPART1','BODYPART2','BODYPART3'],
        'use_bodyparts': ['BODYPART1','BODYPART2','BODYPART3'],
        'skeleton': [['BODYPART1','BODYPART2'], ['BODYPART2','BODYPART3']],
        'anterior_bodyparts': ['BODYPART1'],
        'posterior_bodyparts': ['BODYPART3']})
        
    other = update_dict(kwargs, {
        'video_directory': '',
        'keypoint_colormap': 'autumn',
        'latent_dimension': 10,
        'batch_length': 10000 })
       
    fitting = update_dict(kwargs, {
        'PCA_fitting_num_frames': 100000,
        'PCA_interp_confidence': 0.5,
        'kappa_scan_target_duration': 12,
        'kappa_scan_min': 1e2,
        'kappa_scan_max': 1e12,
        'num_arhmm_scan_iters': 50,
        'num_arhmm_final_iters': 200,
        'num_kpslds_scan_iters': 50,
        'num_kpslds_final_iters': 500,
        'save_every_n_iters': 10})
        
    comments = {
        'video_directory': 'directory with videos from which keypoints were derived (used for crowd movies)',
        'bodyparts': 'used to access columns in the keypoint data',
        'skeleton': 'used for visualization only',
        'use_bodyparts': 'determines the subset of bodyparts to use for modeling and the order in which they are represented',
        'anterior_bodyparts': 'used to initialize heading',
        'posterior_bodyparts': 'used to initialize heading',
        'batch_length': 'data are broken up into batches to parallelize fitting',
        'trans_hypparams': 'transition hyperparameters',
        'ar_hypparams': 'autoregressive hyperparameters',
        'obs_hypparams': 'keypoint observation hyperparameters',
        'cen_hypparams': 'centroid movement hyperparameters',
        'error_estimator': 'parameters to convert neural net likelihoods to error size priors',
        'save_every_n_iters': 'frequency for saving model snapshots during fitting; if 0 only final state is saved', 
        'kappa_scan_target_duration': 'target median syllable duration (in frames) for choosing kappa'
    }
    
    sections = [
        ('ANATOMY', anatomy),
        ('FITTING', fitting),
        ('HYPER PARAMS',hypperams),
        ('OTHER', other)
    ]

    with open(os.path.join(project_directory,'config.yml'),'w') as f: 
        f.write(build_yaml(sections, comments))
                          
        
def check_config_validity(config):
    error_messages = []
    
    # check anatomy
    
    for bodypart in config['use_bodyparts']:
        if not bodypart in config['bodyparts']:
            error_messages.append(
                
                f'ACTION REQUIRED: `use_bodyparts` contains {bodypart} \
                which is not one of the options in `bodyparts`.')
            
    for bodypart in sum(config['skeleton'],[]):
        if not bodypart in config['bodyparts']:
            error_messages.append(
                
                f'ACTION REQUIRED: `skeleton` contains {bodypart} \
                which is not one of the options in `bodyparts`.')
            
    for bodypart in config['anterior_bodyparts']:
        if not bodypart in config['bodyparts']:
            error_messages.append(
                
                f'ACTION REQUIRED: `anterior_bodyparts` contains {bodypart} \
                which is not one of the options in `bodyparts`.')
            
    for bodypart in config['posterior_bodyparts']:
        if not bodypart in config['bodyparts']:
            error_messages.append(
                
                f'ACTION REQUIRED: `posterior_bodyparts` contains {bodypart} \
                which is not one of the options in `bodyparts`.')

    if len(error_messages)>0: 
        print('')
    for msg in error_messages: 
        print(fill(msg, width=70, subsequent_indent='  '), end='\n\n')
            
def load_config(project_directory, check_if_valid=True):
    """
    Load config.yml file from ``project_directory`` and return
    the resulting dict. Optionally check if the config is valid. 
    """
    config_path = os.path.join(project_directory,'config.yml')
    with open(config_path, 'r') as stream:  config = yaml.safe_load(stream)
    if check_if_valid: check_config_validity(config)
    return config
        
def setup_project(project_directory, deeplabcut_config=None, **options):
    """
    Setup a project directory with the following structure
    ```
        project_directory
        ├── config.yml
        ├── figures
        ├── models
        └── stateseqs
    ```
    
    Parameters
    ----------
    project_directory: str 
        Path to the project directory (relative or absolute)
        
    deeplabcut_config: str, default=None
        Path to a deeplabcut config file. Relevant settings will be
        imported and used to initialize the keypoint MoSeq config.
        (overrided by **kwargs)
        
    **options
        Used to initialize config file
    """

    if deeplabcut_config is not None: 
        dlc_options = {}
        with open(deeplabcut_config, 'r') as stream:
            
            dlc_config = yaml.safe_load(stream)
            if dlc_config is None:
                raise RuntimeError(f'{deeplabcut_config} does not exists or is not a valid yaml file')
            if 'multianimalproject' in dlc_config and dlc_config['multianimalproject']:
                raise NotImplementedError('Config initialization from multi-animal deeplabcut projects is not yet supported')
                
            if 'bodyparts' in dlc_config:
                dlc_options['bodyparts'] = dlc_config['bodyparts']
                dlc_options['use_bodyparts'] = dlc_config['bodyparts']
            if 'skeleton' in dlc_config:
                dlc_options['skeleton'] = dlc_config['skeleton']
                
        options = {**dlc_options, **options}
                
    if not os.path.exists(project_directory): 
        os.makedirs(project_directory)
    generate_config(project_directory, **options)
        
    for name in ['figures','models','stateseqs']:
        subdirectory = os.path.join(project_directory,name)
        if not os.path.exists(subdirectory): 
            os.makedirs(subdirectory)
            
    
#

def format_data(coordinates, *, confidence=None, keys=None, 
                batch_length, bodyparts, use_bodyparts, **config):
    """
    Reshapes variable-length time-series of keypoint coordinates
    and neural net `likelihoods` by breaking into batches of fixed 
    length and subsetting/reordering based on ``use_bodyparts``. 
    Batches that include the end of a session are 0-padded. 
    
    Parameters
    ----------
    coordinates: dict
        Keypoint coordinates for a collection of sessions. Values
        must be numpy arrays of shape (T,K,D) where K is the number
        of keypoints and D={2 or 3}. Keys can be any unique str,
        but must the name of a videofile to enable downstream analyses
        such as crowd movies. 
        
    confidence: dict, default=None
        Neural network confidence for a collection of sessions. Values
        must be numpy arrays of shape (T,K) that match the corresponding 
        arrays in ``coordinates``. If ``confidence=None``, all keypoints
        are assigned a confidence of 1.0 by default. 
        
    bodyparts: list of str
        Name of each keypoint. Should have length K corresponding to
        the shape of arrays in ``coordinates``.
        
    use_bodyparts: list of str
        Names of keypoints to use for modeling. Should be a subset of 
        ``bodyparts``.
        
    keys: list, default=None
        Specifies a subset of sessions to include and their order in the 
        final batched array. If ``keys=None``, all sessions will be used 
        and ordered using ``sorted``.
        
    batch_length: int, default=None
        Length of each batch. If ``None``, a length is chosen so that
        no time-series are broken across batched. 
        
    Returns
    -------
    data: dict with the following items
    
        Y: numpy array with shape (n_batches, batch_length, K, D)
            Keypoint coordinates from all sessions broken into batches.
            
        conf: numpy array with shape (n_batches, batch_length, K)
            Neural net confidences from all sessions broken into batches.
        
        mask: numpy array with shape (n_batches, batch_length)
            Binary array where 0 indicates areas of padding.
            
        batch_info: list of tuples (object, int, int)
            The location in ``data_dict`` that each batch came from
            in the form of tuples (key, start, end).
    """    
    
    if keys is None:
        keys = sorted(coordinates.keys())
    if batch_length is None: 
        batch_length = np.max([coordinates[k].shape[0] for k in keys])   
    if confidence is None:
        confidence = {k:np.ones_like(v[...,0]) for k,v in coordinates.items()}

        
    Y,conf,mask,batch_info = [],[],[],[]
    
    for key in keys:
        N,K,D = coordinates[key].shape
        keypoint_ix = np.array([bodyparts.index(bp) for bp in use_bodyparts])
        
        assert K==len(bodyparts), (
            f'`The legth of `bodyparts`` ({len(bodyparts)}) must match the number \
            of keypoints in ``coordinates[{key}]`` ({K})')
        
        for start in range(0,N,batch_length):
            end = min(start+batch_length, N)
            pad_length = batch_length-(end-start)
            padding = np.zeros((pad_length,len(keypoint_ix),D))
            
            Y.append(np.concatenate([coordinates[key][start:end,keypoint_ix],padding],axis=0))
            conf.append(np.concatenate([confidence[key][start:end,keypoint_ix],padding[...,0]],axis=0))
            mask.append(np.hstack([np.ones(end-start),np.zeros(pad_length)]))
            batch_info.append((key,start,end))
            
    return {
        'Y': np.stack(Y), 
        'conf': np.stack(conf),
        'mask': np.stack(mask), 
        'batch_info': batch_info}


def save_pca(pca, project_directory):
    joblib.dump(pca, os.path.join(project_directory,'models','pca.p'))