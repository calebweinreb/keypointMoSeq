from numba import njit, prange
import numpy as np
from jax.config import config
config.update('jax_enable_x64', True)
import jax, jax.numpy as jnp, jax.random as jr
import tensorflow_probability.substrates.jax.distributions as tfd
from itertools import groupby
from functools import partial
na = jnp.newaxis

def masked_mean(X, m, axis=0):
    return (X*m).sum(axis) / (m.sum(axis)+1e-10)

def ppca(key, data, missing, num_components, num_iters=50):
    batch_shape = data.shape[:-1]
    data = data.reshape(-1, data.shape[-1])
    missing = missing.reshape(-1, data.shape[-1])
    N,D = data.shape
    
    observed = ~missing
    total_missing = missing.sum()
    means = masked_mean(data, observed)
    stds = jnp.sqrt(masked_mean((data-means)**2, observed))
    data = (data - means) / stds

    # initial
    C = jr.normal(key, (D, num_components))
    X = data @ C @ jnp.linalg.inv(C.T@C)
    recon = jnp.where(observed, X @ C.T, 0)
    ss = ((recon - data)**2).sum() / (N*D-total_missing)

    for itr in range(num_iters):
        
        # e-step
        data = jnp.where(observed, data, X@C.T)
        Sx = jnp.linalg.inv(jnp.eye(num_components) + C.T@C/ss)
        X = data @ C @ Sx / ss

        # m-step
        C = data.T @ X @ jnp.linalg.pinv(X.T@X + N*Sx)
        recon = jnp.where(observed, X@C.T, 0)
        ss = (((recon-data)**2).sum() + N*(C.T@C*Sx).sum() + total_missing*ss)/(N*D)
    
    U = jnp.linalg.svd(C)[0]
    C = U[:,:num_components]
    vals, vecs = jnp.linalg.eigh(jnp.cov((data @ C).T))
    order = jnp.flipud(jnp.argsort(vals))
    vecs = vecs[:, order]
    vals = vals[order]
    C = C @ vecs
    latents = (data*stds)@C
    return C, means, latents.reshape(*batch_shape, num_components)


@jax.jit
def obs_log_prob(*, Y, mask, x, s, v, h, Cd, sigmasq, **kwargs):
    k,d = Y.shape[-2:]
    Gamma = center_embedding(k)
    Ybar = Gamma @ (pad_affine(x)@Cd.T).reshape(*Y.shape[:-2],k-1,d)
    sqerr = ((Y - affine_transform(Ybar,v,h))**2).sum(-1)
    return -1/2 * sqerr/s/sigmasq - 3/2 * jnp.log(s*sigmasq*jnp.pi)


def center_embedding(k):
    return jnp.linalg.svd(jnp.eye(k) - jnp.ones((k,k))/k)[0][:,:-1]

def stateseq_stats(stateseqs, mask):
    s = np.array(stateseqs.flatten()[mask[...,-stateseqs.shape[-1]:].flatten()>0])
    durations = [sum(1 for i in g) for k,g in groupby(s)]
    usage = np.bincount(s)
    return usage, durations

def merge_data(data_dict, keys=None, batch_length=None):
    if keys is None: keys = sorted(data_dict.keys())
    max_length = np.max([data_dict[k].shape[0] for k in keys])
    if batch_length is None: batch_length = max_length
    else: max_length = int(np.ceil(max_length/batch_length)*batch_length)
      
    def reshape(x):
        x = np.concatenate([x, np.zeros((max_length-x.shape[0],*x.shape[1:]))],axis=0)
        return x.reshape(-1, batch_length, *x.shape[1:])
    
    data = np.concatenate([reshape(data_dict[k]) for k in keys],axis=0)
    mask = np.concatenate([reshape(np.ones(data_dict[k].shape[0])) for k in keys],axis=0)
    keys = [(k,i) for k in keys for i in range(int(len(data_dict[k])/batch_length+1))]
    return data, mask, keys

def ensure_symmetric(X):
    XT = jnp.swapaxes(X,-1,-2)
    return (X+XT)/2

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
    Ytransformed = (Y[...,na,:]*rot_matrix[...,na,:,:]).sum(-1) + v[...,na,:]
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
    Ytransformed = ((Y-v[...,na,:])[...,na,:]*rot_matrix[...,na,:,:]).sum(-1)
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


def get_lags(x, nlags):
    """
    Get lags of a multivariate time series.
    
    Parameters
    ----------  
    nlags: int
        Number of lags
        
    x: ndarray, shape (*dims, t, d)
        Batch of multivariate time series
    
    Returns
    -------
    xlagged: ndarray, shape [*dims, t-nlags, d*nlags]
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


