import jax, jax.numpy as jnp, jax.random as jr
from keypoint_moseq.util import inverse_affine_transform, apply_pca, vector_to_angle
from keypoint_moseq.hdp import sample_hdp_transitions
from keypoint_moseq.distributions import sample_mniw


def initial_location(Y): 
    v = Y.mean(-2)
    return v.at[...,2:].set(0)

def initial_heading(Y, posterior_keypoints, anterior_keypoints):
    posterior_loc = Y[...,posterior_keypoints,:2].mean(-2)
    anterior_loc = Y[...,anterior_keypoints,:2].mean(-2)
    return vector_to_angle(anterior_loc-posterior_loc)

def initial_latents(*, Y, mask, v, h, latent_dim, **kwargs):
    y = inverse_affine_transform(Y,v,h).reshape(*Y.shape[:-2],-1)
    x, C, d = apply_pca(y, mask, latent_dim)
    return x, jnp.vstack([C, d]).T


def initial_ar_params(key, *, num_states, nu_0, S_0, M_0, K_0, **kwargs):
    Ab,Q = jax.vmap(sample_mniw, in_axes=(0,None,None,None,None))(
        jr.split(key, num_states),nu_0,S_0,M_0,K_0)
    return Ab,Q

def initial_transitions(key, *, num_states, alpha, kappa, gamma):
    keys = jr.split(key)
    counts = jnp.zeros((num_states,num_states))
    betas_init = jr.dirichlet(keys[0], jnp.ones(num_states)*gamma/num_states)   
    betas, pi = sample_hdp_transitions(keys[1], counts, betas_init, alpha, kappa, gamma)
    return betas, pi
  