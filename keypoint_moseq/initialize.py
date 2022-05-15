import jax, jax.numpy as jnp, jax.random as jr
from keypoint_moseq.transitions import sample_hdp_transitions, sample_transitions
from keypoint_moseq.distributions import sample_mniw
from keypoint_moseq.util import *
na = jnp.newaxis


def initial_location(*, Y, outliers, **kwargs): 
    m = (outliers==0)[...,na] * jnp.ones_like(Y)
    v = masked_mean(Y, m, axis=-2)
    return v.at[...,2:].set(0)

def initial_heading(posterior_keypoints, anterior_keypoints, *, Y, outliers, **kwargs):
    m = (outliers==0)[...,na] * jnp.ones_like(Y)
    posterior_loc = masked_mean(Y[...,posterior_keypoints,:2], 
                                m[...,posterior_keypoints,:2], axis=-2)
    anterior_loc = masked_mean(Y[...,anterior_keypoints,:2],
                               m[...,anterior_keypoints,:2], axis=-2)
    return vector_to_angle(anterior_loc-posterior_loc)

def initial_latents(key, *, Y, outliers, mask, v, h, latent_dim, **kwargs):
    y = inverse_affine_transform(Y,v,h).reshape(*Y.shape[:-2],-1)
    missing = jnp.repeat(outliers,Y.shape[-1],axis=-1)>0
    components, means, latents = ppca(key, y, missing, latent_dim)
    Gamma = center_embedding(Y.shape[-2])
    Cd = jnp.hstack([components, means[:,None]])
    Cd = jnp.kron(Gamma.T, jnp.eye(Y.shape[-1]))@Cd
    return latents, Cd

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
  