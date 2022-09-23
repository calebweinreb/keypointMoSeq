import jax, jax.numpy as jnp, jax.random as jr
from keypoint_moseq.transitions import sample_hdp_transitions, sample_transitions
from keypoint_moseq.distributions import sample_mniw
from keypoint_moseq.util import *
from sklearn.decomposition import PCA
na = jnp.newaxis

'''
def initial_latents(*, Y, mask, v, h, latent_dim, num_samples=100000, **kwargs):
    n,t,k,d = Y.shape
    y = center_embedding(k).T @ inverse_affine_transform(Y,v,h) 
    yflat = y.reshape(t*n, (k-1)*d)
    ysample = np.array(yflat)[np.random.choice(t*n,num_samples)]
    pca = PCA(n_components=latent_dim, whiten=True).fit(ysample)
    latents = jnp.array(pca.transform(yflat).reshape(n,t,latent_dim))
    Cd = jnp.array(jnp.hstack([pca.components_.T, pca.mean_[:,na]]))
    return latents, Cd, pca
'''

def initial_latents(*, Y, mask, v, h, latent_dim, num_samples=100000, whiten=True, pca=None, **kwargs):
    n,t,k,d = Y.shape
    y = center_embedding(k).T @ inverse_affine_transform(Y,v,h) 
    yflat = y.reshape(t*n, (k-1)*d)
    ysample = np.array(yflat)[np.random.choice(t*n,num_samples)]
    if pca is None: pca = PCA(n_components=latent_dim).fit(ysample)
    latents_flat = jnp.array(pca.transform(yflat))[:,:latent_dim]
    Cd = jnp.array(jnp.hstack([pca.components_.T, pca.mean_[:,na]]))
    
    if whiten:
        cov = jnp.cov(latents_flat[mask.flatten()>0].T)
        L = jnp.linalg.cholesky(cov)
        Linv = jnp.linalg.inv(L)
        latents_flat = latents_flat @ Linv.T
        Cd = Cd.at[:,:-1].set(Cd[:,:-1] @ L)
                             
    latents = latents_flat.reshape(n,t,latent_dim)    
    return latents, Cd, pca

def initial_location(*, Y, **kwargs): 
    return Y.mean(-2).at[...,2:].set(0)

def initial_heading(posterior_keypoints, anterior_keypoints, *, Y, **kwargs):
    posterior_loc = Y[..., posterior_keypoints,:2].mean(-2) 
    anterior_loc = Y[..., anterior_keypoints,:2].mean(-2) 
    return vector_to_angle(anterior_loc-posterior_loc)


def initial_ar_params(key, *, num_states, nu_0, S_0, M_0, K_0, **kwargs):
    Ab,Q = jax.vmap(sample_mniw, in_axes=(0,na,na,na,na))(
        jr.split(key, num_states),nu_0,S_0,M_0,K_0)
    return Ab,Q

def initial_hdp_transitions(key, *, num_states, alpha, kappa, gamma):
    keys = jr.split(key)
    counts = jnp.zeros((num_states,num_states))
    betas_init = jr.dirichlet(keys[0], jnp.ones(num_states)*gamma/num_states)   
    betas, pi = sample_hdp_transitions(keys[1], counts, betas_init, alpha, kappa, gamma)
    return betas, pi

def initial_transitions(key, *, num_states, alpha, kappa):
    keys = jr.split(key)
    counts = jnp.zeros((num_states,num_states))
    pi = sample_transitions(keys[1], counts, alpha, kappa)
    return pi
  