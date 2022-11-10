from jax.tree_util import tree_map
import jax.numpy as jnp
import numpy as np
import joblib
import tqdm
import yaml
import os
import cv2
import tqdm
import re
import pandas as pd
from datetime import datetime
from textwrap import fill
from vidio.read import OpenCVReader
from keypoint_moseq.util import unwrap_stateseqs, to_np

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
        'error_estimator': {'slope':1, 'intercept':1},
        'obs_hypparams': {'sigmasq_0':0.1, 'sigmasq_C':.1, 'nu_sigma':1e5, 'nu_s':5},
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
        'verbose':True,
        'conf_pseudocount': 1e-3,
        'video_directory': '',
        'keypoint_colormap': 'autumn',
        'latent_dimension': 10,
        'whiten': True,
        'batch_length': 10000 })
       
    fitting = update_dict(kwargs, {
        'added_noise_level': 0.1,
        'PCA_fitting_num_frames': 1000000,
        'conf_threshold': 0.5,
#         'kappa_scan_target_duration': 12,
#         'kappa_scan_min': 1e2,
#         'kappa_scan_max': 1e12,
#         'num_arhmm_scan_iters': 50,
#         'num_arhmm_final_iters': 200,
#         'num_kpslds_scan_iters': 50,
#         'num_kpslds_final_iters': 500
    })
        
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
        'kappa_scan_target_duration': 'target median syllable duration (in frames) for choosing kappa',
        'whiten': 'whether to whiten principal components; used to initialize the latent pose trajectory `x`',
        'conf_threshold': 'used to define outliers for interpolation when the model is initialized',
        'conf_pseudocount': 'pseudocount used regularize neural network confidences',
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
                f'ACTION REQUIRED: `use_bodyparts` contains {bodypart} '
                'which is not one of the options in `bodyparts`.')
            
    for bodypart in sum(config['skeleton'],[]):
        if not bodypart in config['bodyparts']:
            error_messages.append(
                f'ACTION REQUIRED: `skeleton` contains {bodypart} '
                'which is not one of the options in `bodyparts`.')
            
    for bodypart in config['anterior_bodyparts']:
        if not bodypart in config['bodyparts']:
            error_messages.append(
                f'ACTION REQUIRED: `anterior_bodyparts` contains {bodypart} '
                'which is not one of the options in `bodyparts`.')
            
    for bodypart in config['posterior_bodyparts']:
        if not bodypart in config['bodyparts']:
            error_messages.append(     
                f'ACTION REQUIRED: `posterior_bodyparts` contains {bodypart} '
                'which is not one of the options in `bodyparts`.')

    if len(error_messages)>0: 
        print('')
    for msg in error_messages: 
        print(fill(msg, width=70, subsequent_indent='  '), end='\n\n')
            
def load_config(project_directory, check_if_valid=True):
    """
    Load config.yml from ``project_directory`` and return
    the resulting dict. Optionally check if the config is valid. 
    """
    config_path = os.path.join(project_directory,'config.yml')
    with open(config_path, 'r') as stream:  config = yaml.safe_load(stream)
    if check_if_valid: check_config_validity(config)
    return config

def update_config(project_directory, **kwargs):
    """
    Update config.yml from ``project_directory`` to include
    all the key/value pairs in **kwargs.
    """
    config = load_config(project_directory, check_if_valid=False)
    config.update(kwargs)
    generate_config(project_directory, **config)
    
        
def setup_project(project_directory, deeplabcut_config=None, 
                  overwrite=False, **options):
    """
    Setup a project directory with the following structure
    ```
        project_directory
        ├── config.yml
        ├── misc
        ├── figures
        └── models
    ```
    
    Parameters
    ----------
    project_directory: str 
        Path to the project directory (relative or absolute)
        
    deeplabcut_config: str, default=None
        Path to a deeplabcut config file. Relevant settings will be
        imported and used to initialize the keypoint MoSeq config.
        (overrided by **kwargs)
        
    overwrite: bool, default=False
        Overwrite any config.yml that already exists at the path
        ``[project_directory]/config.yml``
        
    **options
        Used to initialize config file
    """

    if os.path.exists(project_directory) and not overwrite:
        print(fill(f'The directory `{project_directory}` already exists. Use `overwrite=True` or pick a different name for the project directory'))
        return
        
    if deeplabcut_config is not None: 
        dlc_options = {}
        with open(deeplabcut_config, 'r') as stream:           
            dlc_config = yaml.safe_load(stream)
            
            if dlc_config is None:
                raise RuntimeError(
                    f'{deeplabcut_config} does not exists or is not a valid yaml file')
                
            if 'multianimalproject' in dlc_config and dlc_config['multianimalproject']:
                raise NotImplementedError(
                    'Config initialization from multi-animal deeplabcut '
                    'projects is not yet supported')
                
            dlc_options['bodyparts'] = dlc_config['bodyparts']
            dlc_options['use_bodyparts'] = dlc_config['bodyparts']
            dlc_options['skeleton'] = dlc_config['skeleton']
            dlc_options['video_directory'] = os.path.join(dlc_config['project_path'],'videos')
                
        options = {**dlc_options, **options}
    
    if not os.path.exists(project_directory):
        os.makedirs(project_directory)
    generate_config(project_directory, **options)
        
    for name in ['figures','models','misc']:
        subdirectory = os.path.join(project_directory,name)
        if not os.path.exists(subdirectory): 
            os.makedirs(subdirectory)
            
    
#

def format_data(coordinates, *, confidences=None, keys=None, 
                batch_length, bodyparts, use_bodyparts, batch_overlap=30,
                conf_pseudocount=1e-3, added_noise_level=0.1, **kwargs):
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
        
    confidences: dict, default=None
        Neural network confidences for a collection of sessions. Values
        must be numpy arrays of shape (T,K) that match the corresponding 
        arrays in ``coordinates``. 
        
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
        
    conf_pseudocount: float, default=1e-3
        Pseudocount neural network confidences.
    
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
            If no input is provided for ``confidences``, will be ``None``. 
            Confidences are increased by ``conf_pseudocount``.
        
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
    if confidences is None:
        confidences = {k:np.ones_like(v[...,0]) for k,v in coordinates.items()}

        
    Y,conf,mask,batch_info = [],[],[],[]
    
    for key in keys:
        N,K,D = coordinates[key].shape
        keypoint_ix = np.array([bodyparts.index(bp) for bp in use_bodyparts])
        
        assert K==len(bodyparts), fill(
            f'`The legth of `bodyparts`` ({len(bodyparts)}) must match the number \
            of keypoints in ``coordinates[{key}]`` ({K})')
        
        for start in range(0,N,batch_length):
            
            end = min(start+batch_length+batch_overlap, N)
            pad_length = batch_length+batch_overlap-(end-start)
            padding = np.zeros((pad_length,len(keypoint_ix),D))
            batch_info.append((key,start,end))
            
            mask.append(np.hstack([
                np.ones(end-start),
                np.zeros(pad_length)]))
            
            Y.append(np.concatenate([
                coordinates[key][start:end,keypoint_ix],
                padding],axis=0))
            
            if confidences is not None: 
                conf.append(np.concatenate([
                    confidences[key][start:end,keypoint_ix],
                    padding[...,0]],axis=0))

    Y = np.stack(Y)
    mask = np.stack(mask)
    conf = None if confidences is None else np.stack(conf)+conf_pseudocount
    if added_noise_level>0 : Y += np.random.uniform(-added_noise_level,added_noise_level,Y.shape)
    return {'mask':mask, 'Y':Y, 'conf':conf}, batch_info


def save_pca(pca, project_directory, pca_path=None):
    if pca_path is None: 
        pca_path = os.path.join(project_directory,'misc','pca.p')
    joblib.dump(pca, pca_path)
    
def load_pca(project_directory, pca_path=None):
    if pca_path is None:
        pca_path = os.path.join(project_directory,'misc','pca.p')
        assert os.path.exists(pca_path), fill(
            f'No PCA model found at {pca_path}. Either save '
            'a new model or specify an alternative path')
    return joblib.load(pca_path)


def load_last_model(project_directory):
    pattern = re.compile(r'(\d{4}_\d{1,2}_\d{1,2}-\d{2}_\d{2}_\d{2})')
    model_dir = os.path.join(project_directory,'models')
    paths = list(filter(lambda p: pattern.search(p), os.listdir(model_dir)))
    assert len(paths)>0, fill(
        f'There are no files in {model_dir} that contain  '
        'a date string with format %Y_%m_%d-%H_%M_%S')
    
    last_model_name = sorted(paths, key=lambda p: datetime.strptime(
            pattern.search(p).group(), '%Y_%m_%d-%H_%M_%S'))[-1]
    save_dict = joblib.load(os.path.join(model_dir,last_model_name))
    return save_dict,last_model_name


def save_model(model, data, history, batch_info, 
               iteration, save_path, save_history=True, 
               save_states=True, save_data=True):
    
    save_dict = {
        'batch_info': batch_info,
        'iteration' : iteration,
        'hypparams' : to_np(model['hypparams']),
        'params'    : to_np(model['params']), 
        'key'       : np.array(model['key']),
        'name'      : os.path.splitext(os.path.basename(save_path))[0]}

    if save_data: save_dict.update(to_np(data))
    if save_states or save_data: save_dict['mask'] = np.array(data['mask'])
    if save_history: save_dict['history'] = {k:np.array(v) for k,v in history.items()}
        
    if save_states: 
        save_dict['states'] = to_np(model['states'])
        save_dict['syllable_seqs'] = unwrap_stateseqs(
            np.array(model['states']['z']), 
            np.array(data['mask']), batch_info)
        
    joblib.dump(save_dict, save_path)




def load_keypoints_from_deeplabcut_file(filepath, *, bodyparts, **kwargs):
    ext = os.path.splitext(filepath)[1]
    assert ext in ['.csv','.h5']
    if ext=='.h5': df = pd.read_hdf(filepath)
    if ext=='.csv': df = pd.read_csv(filepath)
        
    dlc_bodyparts = list(zip(*df.columns.to_list()))[1][::3]
    assert dlc_bodyparts==tuple(bodyparts), fill(
        f'{os.path.basename(filepath)} contains bodyparts'
        f'\n\n{dlc_bodyparts}\n\nbut expected\n\n{bodyparts}')
    
    arr = df.to_numpy().reshape(-1, len(bodyparts), 3)
    coordinates,confidences = arr[:,:,:-1],arr[:,:,-1]
    return coordinates,confidences


def load_keypoints_from_deeplabcut_list(paths, **kwargs): 
    coordinates,confidences = {},{}
    for filepath in tqdm.tqdm(paths, desc='Loading from deeplabcut'):
        coordinates[filepath],confidences[filepath] = \
            load_keypoints_from_deeplabcut_file(filepath, **kwargs)
    return coordinates,confidences
        
    
def load_keypoints_from_deeplabcut(*, video_directory, directory=None, **kwargs):
    if directory is None:
        directory = video_directory
        print(fill(f'Searching in {directory}. Use the ``directory`` '
              'argument to specify another search location'))
    filepaths = [
        os.path.join(directory,f) 
        for f in os.listdir(directory)
        if os.path.splitext(f)[1] in ['.csv','.h5']]
    return load_keypoints_from_deeplabcut_list(filepaths, **kwargs)


