import joblib
import os
import numpy as np
import tqdm
from textwrap import wrap
from datetime import datetime
from keypoint_moseq.model.gibbs import resample_model
from keypoint_moseq.model.initialize import initialize_model
from keypoint_moseq.project.viz import plot_progress
from keypoint_moseq.util import to_jax, to_np, get_durations
from keypoint_moseq.project.io import save_model




    
    
def update_history(history, iteration, *, Y, mask, states, params, **kwargs): 
    history['iteration'].append(iteration)
    for k,v in params.items(): 
        if k in history: history[k].append(np.array(v))        
    for k,v in states.items(): 
        if k in history: history[k].append(np.array(v))
    if 'median_duration' in history:
        durations = get_durations(*to_np((states['z'], mask)))
        history['median_duration'].append(np.median(durations))
    return history



def fit_model(model,
              data,
              batch_info,
              start_iter=0,
              history=None,
              verbose=True,
              num_iters=50,
              ar_only=False,
              history_variables=['z'],
              history_every_n_iters=5,
              plot_every_n_iters=10,   
              project_directory=None,
              save_name=None,
              save_path=None,
              save_every_n_iters=10,
              save_history=True,
              save_states=True,
              save_data=True,
              **kwargs):
    
    
    if save_every_n_iters>0:
        if save_path is None:
            assert project_directory, wrap(
                'To save the model during fitting, provide '
                'a ``project_directory`` or ``save_path``. '
                'Otherwise turn off model saving with ``save_every_n_iters=0``')
            if save_name is None: 
                date_str = str(datetime.now().strftime('%Y_%m_%d-%H_%M_%S'))
                suffix = 'ARHMM' if ar_only else 'KPSLDS'
                save_name = f'{date_str}-{suffix}.p'
            else: save_name = os.path.splitext(save_name)[0]+'.p'
            save_path = os.path.join(project_directory,'models',save_name)
    
    if history is None:
        history_variables += ['iteration', 'median_duration']
        history = {k:[] for k in history_variables}
    else: history = {k:list(v) for k,v in history.items()}
    
    for iteration in tqdm.trange(start_iter, num_iters):
        if history_every_n_iters>0 and (iteration%history_every_n_iters)==0:
            history = update_history(history, iteration, **model, **data)
            
        if plot_every_n_iters>0 and (iteration%plot_every_n_iters)==0:
            plot_progress(history=history, iteration=iteration, **data, **model)

        if save_every_n_iters>0 and (iteration%save_every_n_iters)==0:
            if verbose: print(wrap(f'Saving model to {save_path}'))
            save_model(model, data, history, batch_info, iteration, 
                       save_path, save_history=save_history, 
                       save_states=save_states, save_data=save_data)
            
        try: model = resample_model(data, **model, ar_only=ar_only)
        except KeyboardInterrupt: break
    
    return model, {k:np.array(v) for k,v in history.items()}
    
    
def resume_fitting(model, *, batch_info, iteration, history, 
                   name, Y, conf, mask, project_directory, 
                   ar_only=False, **kwargs):
    
    data = to_jax({'Y':Y, 'conf':conf, 'mask':mask})
   
    if 'ARHMM' in name and not ar_only:
        new_name = name.replace('ARHMM','KPSLDS')
        print(wrap(f'Switching model name from {name} to {new_name}'))
        name = new_name
    
    return fit_model(
        model, data, batch_info, ar_only=ar_only, 
        project_directory=project_directory, save_name=name, 
        history=history, start_iter=iteration+1, **kwargs)

