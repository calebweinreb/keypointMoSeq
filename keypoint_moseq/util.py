from numba import njit, prange
import numpy as np
from jax.config import config
config.update('jax_enable_x64', True)
import jax, jax.numpy as jnp, jax.random as jr
import tensorflow_probability.substrates.jax.distributions as tfd
from itertools import groupby
from functools import partial
na = jnp.newaxis

def expected_keypoints(*, Y, v, h, x, Cd, **kwargs):
    k,d = Y.shape[-2:]
    Gamma = center_embedding(k)
    Ybar = Gamma @ (pad_affine(x)@Cd.T).reshape(*Y.shape[:-2],k-1,d)
    Yexp = affine_transform(Ybar,v,h)
    return Yexp


def pca_transform(components, means, *, Y, v, h, **kwargs):
    y = inverse_affine_transform(Y,v,h).reshape(*Y.shape[:-2],-1)
    return (y-means) @ components

def interpolate_keypoints(keypoints, outliers, axis=1):
    keypoints = np.moveaxis(keypoints, axis, 0)
    outliers = np.moveaxis(outliers, axis, 0)
    init_shape = keypoints.shape
    outliers = np.repeat(outliers[...,None],init_shape[-1],axis=-1)
    keypoints = keypoints.reshape(init_shape[0],-1)
    outliers = outliers.reshape(init_shape[0],-1)
    for i in range(keypoints.shape[1]):
        keypoints[:,i] = np.interp(
            np.arange(init_shape[0]), 
            np.nonzero(~outliers[:,i])[0],
            keypoints[:,i][~outliers[:,i]])
    return np.moveaxis(keypoints.reshape(init_shape),0,axis)



'''
def ppca(key, data, missing, num_components, num_iters=50, whiten=True):
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
    
    if whiten:
        Sigma = jnp.cov(latents.T)
        L = jnp.linalg.cholesky(Sigma)
        latents = jnp.linalg.solve(L, latents.T).T
        C = jnp.linalg.solve(L, C.T).T
    return C, means, latents.reshape(*batch_shape, num_components)


def whiten(x):
    shape,x = x.shape, x.reshape(-1,x.shape[-1])
    mu = x[mask.flatten()>0].mean(0)
    Sigma = jnp.cov(x[mask.flatten()>0].T)
    L = jnp.linalg.cholesky(Sigma)
    x = jnp.linalg.solve(L, (x-mu).T).T
    return x.reshape(shape), L
'''

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
        
    def reshape(x):
        padding = (-x.shape[0])%batch_length
        x = np.concatenate([x, np.zeros((padding,*x.shape[1:]))],axis=0)
        return x.reshape(-1, batch_length, *x.shape[1:])
    
    data = np.concatenate([reshape(data_dict[k]) for k in keys],axis=0)
    mask = np.concatenate([reshape(np.ones(data_dict[k].shape[0])) for k in keys],axis=0)
    keys = [(k,i) for k in keys for i in range(int(np.ceil(len(data_dict[k])/batch_length)))]
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

def gaussian_log_prob(x, mu, sigma_inv):
    return (-((mu-x)[...,na,:]*sigma_inv*(mu-x)[...,:,na]).sum((-1,-2))/2
            +jnp.log(jnp.linalg.det(sigma_inv))/2)


def latent_log_prob(*, x, z, Ab, Q, **kwargs):
    Qinv = jnp.linalg.inv(Q)
    Qdet = jnp.linalg.det(Q)
    
    nlags = Ab.shape[2]//Ab.shape[1]
    x_lagged = get_lags(x, nlags)
    x_pred = (Ab[z] @ pad_affine(x_lagged)[...,na])[...,0]
    
    d = x_pred - x[:,nlags:]
    return (-(d[...,na,:]*Qinv[z]*d[...,:,na]).sum((2,3))/2
            -jnp.log(Qdet[z])/2  -jnp.log(2*jnp.pi)*Q.shape[-1]/2)

def stateseq_log_prob(*, z, pi, **kwargs):
    return jnp.log(pi[z[:,:-1],z[:,1:]])

def scale_log_prob(*, s, nu_s, s_0, **kwargs):
    return -nu_s*s_0 / s / 2 - (1+nu_s/2)*jnp.log(s)
    
def location_log_prob(*, v, sigmasq_loc):
    d = v[:,:-1]-v[:,1:]
    return (-(d**2).sum(-1)/sigmasq_loc/2 
            -v.shape[-1]/2*jnp.log(sigmasq_loc*2*jnp.pi))

def obs_log_prob(*, Y, x, s, v, h, Cd, sigmasq, **kwargs):
    k,d = Y.shape[-2:]
    Gamma = center_embedding(k)
    Ybar = Gamma @ (pad_affine(x)@Cd.T).reshape(*Y.shape[:-2],k-1,d)
    sqerr = ((Y - affine_transform(Ybar,v,h))**2).sum(-1)
    return (-1/2 * sqerr/s/sigmasq - d/2 * jnp.log(2*s*sigmasq*jnp.pi))

@jax.jit
def log_joint_likelihood(*, Y, mask, x, s, v, h, z, pi, Ab, Q, Cd, sigmasq, sigmasq_loc, nu_s, s_0, **kwargs):
    nlags = Ab.shape[2]//Ab.shape[1]
    return {
        'Y': (obs_log_prob(Y=Y, x=x, s=s, v=v, h=h, Cd=Cd, sigmasq=sigmasq)*mask[:,:,na]).sum(),
        'x': (latent_log_prob(x=x, z=z, Ab=Ab, Q=Q)*mask[:,nlags:]).sum(),
        'z': (stateseq_log_prob(z=z, pi=pi)*mask[:,nlags+1:]).sum(),
        'v': (location_log_prob(v=v, sigmasq_loc=sigmasq_loc)*mask[:,1:]).sum(),
        's': (scale_log_prob(s=s, nu_s=nu_s, s_0=s_0)*mask[:,:,na]).sum()}

