from numba import njit, prange
import numpy as np
from jax.config import config
config.update("jax_enable_x64", True)
import jax, jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd
from itertools import groupby
from functools import partial

def stateseq_stats(stateseqs, mask):
    s = np.array(stateseqs.flatten()[mask[...,-stateseqs.shape[-1]:].flatten()>0])
    durations = [sum(1 for i in g) for k,g in groupby(s)]
    usage = np.bincount(s)
    return usage, durations

def merge_data(data_dict, keys=None):
    if keys is None: keys = sorted(data_dict.keys())
    max_length = np.max([data_dict[k].shape[0] for k in keys])
    pad = lambda x: np.concatenate([x, np.zeros((max_length-x.shape[0],*x.shape[1:]))],axis=0)
    data = np.stack([pad(data_dict[k]) for k in keys],axis=0)
    mask = np.stack([pad(np.ones(data_dict[k].shape[0])) for k in keys],axis=0)
    return data, mask, keys


def vector_to_angle(V):
    """Convert 2D vectors to angles in [-pi, pi]. The vector (1,0)
    corresponds to angle of 0. If V is n-dinmensional, the first
    n-1 dimensions are treated as batch dims. 

    Parameters
    ----------
    V: ndarray, shape (*dims, d) where d >= 2

    Returns
    -------
    angles: ndarray, shape (*dims)
    
    """
    y,x = V[...,1],V[...,0]+1e-10
    angles = (jnp.arctan(y/x)+(x>0)*jnp.pi)%(2*jnp.pi)-jnp.pi
    return angles

    
def angle_to_rotation_matrix(h, keypoint_dim=3):
    """Create rotation matrices from an array of angles. If
    `keypoint_dim > 2` then rotation is performed in the 
    first two dims and the remaing axes are fixed.

    Parameters
    ----------
    h: ndarray, shape (*dims)
        Angles (in radians)

    Returns
    -------
    m: ndarray, shape (*dims, keypoint_dim, keypoint_dim)
        Stacked rotation matrices 
        
    keypoint_dim: int, default=3
        Dimension of each rotation matrix
    
    """
    m = jnp.tile(jnp.eye(keypoint_dim), (*h.shape,1,1))
    m = m.at[...,0,0].set(jnp.cos(h))
    m = m.at[...,1,1].set(jnp.cos(h))
    m = m.at[...,0,1].set(-jnp.sin(h))
    m = m.at[...,1,0].set(jnp.sin(h))
    return m


@jax.jit
def affine_transform(Y, v, h):
    """
    Apply the following affine transform, batching over keypoints.
    
    .. math::
        Y \mapsto R(h) @ Y + v

    Parameters
    ----------   
    Y: ndarray, shape (*dims, k, d) where d >= 2
        Batches of k keypoints in d-dim embedding space
        
    v: ndarray, shape (*dims, d)
        Translations
        
    h: ndarray, shape (*dims)
        Angles in radians
          
    Returns
    -------
    Ytransformed: ndarray, shape (*dims, k, d)
        
    """
    rot_matrix = angle_to_rotation_matrix(h, keypoint_dim=Y.shape[-1])
    Ytransformed = (Y[...,None,:]*rot_matrix[...,None,:,:]).sum(-1) + v[...,None,:]
    return Ytransformed

@jax.jit
def inverse_affine_transform(Y, v, h):
    """
    Apply the following affine transform, batching over keypoints.
    
    .. math::
        Y \mapsto R(-h) @ (Y - v)

    Parameters
    ----------   
    Y: ndarray, shape (*dims, k, d) where d >= 2
        Batches of k keypoints in d-dim embedding space
        
    v: ndarray, shape (*dims, d)
        Translations
        
    h: ndarray, shape (*dims)
        Angles in radians
          
    Returns
    -------
    Ytransformed: ndarray, shape (*dims, k, d)
        
    """
    rot_matrix = angle_to_rotation_matrix(-h, keypoint_dim=Y.shape[-1])
    Ytransformed = ((Y-v[...,None,:])[...,None,:]*rot_matrix[...,None,:,:]).sum(-1)
    return Ytransformed
    

def pad_affine(x):
    """
    Pad with 1's to enable affine transform with matrix multiplication.
    Padding is appended and the end of the last dimension.
    """
    padding = jnp.ones((*x.shape[:-1],1))
    xpadded = jnp.concatenate((x,padding),axis=-1)
    return xpadded

@njit
def count_transitions(num_states, mask, stateseqs):
    """
    Count all transitions in `stateseqs` where the start and end
    states both have `mask>0`. The last dim of `stateseqs` indexes time. 
    """    
    counts = np.zeros((num_states,num_states))
    for i in prange(mask.shape[0]):
        for j in prange(mask.shape[1]-1):
            if mask[i,j]>0 and mask[i,j+1]>0:
                counts[stateseqs[i,j],stateseqs[i,j+1]] += 1
    return counts


def add_lags(x, nlags):
    """
    Append lags to multivariate time series.
    
    Parameters
    ----------  
    nlags: int
        Number of lags
        
    x: ndarray, shape (*dims, t, d)
        Batch of multivariate time series
    
    Returns
    -------
    xlagged: ndarray, shape (*dims, t-nlags, d*nlags)
        Array formed by concatenating lags of x along the last dim.
        The last row of `xlagged` is `x[...,-nlags,:],...,x[...,-2,:]`
    """
    lags = [jnp.roll(x,t,axis=-2) for t in range(1,nlags+1)]
    return jnp.concatenate(lags[::-1],axis=-1)[...,nlags:,:]


def ar_to_lds(As, bs, Qs, Cs):
    k,l = As.shape[1],As.shape[2]//As.shape[1]

    A_ = jnp.zeros((As.shape[0],k*l,k*l))
    A_ = A_.at[:,:-k,k:].set(jnp.eye(k*(l-1)))
    A_ = A_.at[:,-k:].set(As)
    
    Q_ = jnp.zeros((Qs.shape[0],k*l,k*l))
    Q_ = Q_.at[:,:-k,:-k].set(jnp.eye(k*(l-1))*1e-2)
    Q_ = Q_.at[:,-k:,-k:].set(Qs)
    
    b_ = jnp.zeros((bs.shape[0],k*l))
    b_ = b_.at[:,-k:].set(bs)
    
    C_ = jnp.zeros((Cs.shape[0],k*l))
    C_ = C_.at[:,-k:].set(Cs)
    return A_, b_, Q_, C_


def apply_pca(X, mask, n_components, max_samples=100000):
    initial_shape = X.shape[:-1]
    X = np.array(X.reshape(-1,X.shape[-1]))
    X_use = X[np.array(mask.flatten())>0]
    if X_use.shape[0] > max_samples:
        ix = np.random.permutation(X_use.shape[0])
        X_use = X_use[ix[:max_samples]]

    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components).fit(X_use)
    X_transformed = jnp.array(pca.transform(X).reshape(*initial_shape,-1))
    components = jnp.array(pca.components_)
    mean = jnp.array(pca.mean_)
    return X_transformed, components, mean
