from jax.config import config
config.update("jax_enable_x64", True)
import jax, jax.numpy as jnp, jax.random as jr
from keypoint_moseq.util import *
from keypoint_moseq.distributions import *
from keypoint_moseq.hdp import sample_hdp_transitions
from keypoint_moseq.kalman import kalman_sample


@jax.jit
def resample_latents(key, *, Y, mask, v, h, z, s, Cd, sigmasq, Ab, Q, **kwargs):
    k,nlags,n = Ab.shape[1],Ab.shape[2]//Ab.shape[1],Y.shape[0]
    ys = inverse_affine_transform(Y,v,h).reshape(*Y.shape[:-2],-1)
    A, B, Q, C, D = *ar_to_lds(Ab[...,:-1],Ab[...,-1],Q,Cd[...,:-1]),Cd[...,-1]
    R = jnp.repeat(s*sigmasq,Y.shape[-1],axis=-1)[:,nlags:]
    mu0,S0 = jnp.zeros((n,k*nlags)),jnp.repeat(jnp.eye(k*nlags)[None]*10,n,axis=0)
    xs = jax.vmap(kalman_sample, in_axes=(0,0,0,0,0,0,0,0,None,None,0))(
        jr.split(key,ys.shape[0]), ys[:,nlags:], mask[:,nlags:], mu0, S0, A[z], B[z], Q[z], C, D, R)
    return jnp.concatenate([mu0.reshape(n,nlags,k),xs[:,:,-k:]],axis=1)

@jax.jit
def resample_heading(key, *, Y, v, x, s, Cd, sigmasq, **kwargs):
    Cx_tild = (pad_affine(x)@Cd.T).reshape(Y.shape)[...,None]
    Ycen = (Y - v[...,None,:])[...,None,:]
    S = (Cx_tild*Ycen/(s*sigmasq)[...,None,None]).sum(-3)
    kappa_cos = S[...,0,0]+S[...,1,1]
    kappa_sin = S[...,0,1]-S[...,1,0]
    theta = vector_to_angle(jnp.stack([kappa_cos,kappa_sin],axis=-1))
    kappa = jnp.sqrt(kappa_cos**2 + kappa_sin**2)
    return sample_vonmises(key, theta, kappa)

    
@jax.jit 
def resample_location(key, *, mask, Y, h, x, s, Cd, sigmasq, sigmasq_loc, **kwargs):
    d = Y.shape[-1]
    gammasq = 1/(1/(s*sigmasq)).sum(-1)
    Cx_tild = (pad_affine(x)@Cd.T).reshape(Y.shape)
    rot_matrix = angle_to_rotation_matrix(-h, keypoint_dim=d)
    mu = ((Y - (rot_matrix[...,None,:,:]*Cx_tild[...,None,:]).sum(-1)) \
          *(gammasq[...,None]/(s*sigmasq))[...,None]).sum(-2)

    m0, S0 = mu[:,0], gammasq[:,0][...,None,None]*jnp.eye(d)
    As = jnp.tile(jnp.eye(d), (*mask.shape,1,1))
    Bs = jnp.zeros((*mask.shape,d))
    Qs = jnp.tile(jnp.eye(d), (*mask.shape,1,1))*sigmasq_loc
    C,D,Rs = jnp.eye(d),jnp.zeros(d),gammasq[...,None]*jnp.ones(d)
    return jax.vmap(kalman_sample, in_axes=(0,0,0,0,0,0,0,0,None,None,0))(
        jr.split(key,mask.shape[0]), mu, mask, m0, S0, As, Bs, Qs, C, D, Rs)

@jax.jit
def resample_obs_params(key, *, Y, mask, v, h, x, s, K_0, M_0, nu_0, sigmasq_0, **kwargs):
    k,d = Y.shape[-2:]
    K_0_inv = inv_psd(K_0)
    x = pad_affine(x).reshape(-1,x.shape[-1]+1)
    mask = mask.flatten()
    
    def resample_obs_k(key, y, s, M_k):
        S_xx = (x[:,:,None]*x[:,None,:]/s[:,None,None]*mask[:,None,None]).sum(0)
        S_yx = (y[:,:,None]*x[:,None,:]/s[:,None,None]*mask[:,None,None]).sum(0)
        S_yy = (y[:,:,None]*y[:,None,:]/s[:,None,None]*mask[:,None,None]).sum(0)

        nu_n = nu_0 + d*x.shape[0]
        K_n = inv_psd(K_0_inv+S_xx)
        M_n = (M_k@K_0_inv + S_yx)@K_n
        sigmasq_n = jnp.trace(
              S_yy + nu_0*sigmasq_0 
            + M_k@K_0_inv@M_k.T 
            - M_n@(K_0_inv+S_xx)@M_n.T)/nu_n
        
        sigmasq = sample_scaled_inv_chi2(key, nu_n, sigmasq_n)
        Cd = sample_mn(key, M_n, sigmasq*jnp.eye(d), K_n)
        return Cd, sigmasq
    
    Cd, sigmasq = jax.vmap(resample_obs_k, in_axes=(0,1,1,0))(
        jr.split(key,k), inverse_affine_transform(Y,v,h).reshape(-1,k,d),
        s.reshape(-1,k), M_0.reshape(k,d,x.shape[-1]))
    return jnp.concatenate(Cd,axis=0), sigmasq


def _ar_log_likelihood(x, params):
    Ab, Q = params
    nlags = Ab.shape[-1]//Ab.shape[-2]
    mu = pad_affine(add_lags(x, nlags))@Ab.T
    return tfd.MultivariateNormalFullCovariance(mu, Q).log_prob(x[...,nlags:,:])


@jax.jit
def resample_stateseqs(key, *, x, mask, Ab, Q, pi, **kwargs):
    nlags = Ab.shape[2]//Ab.shape[1]
    log_likelihoods = jax.lax.map(partial(_ar_log_likelihood,x), (Ab, Q))
    stateseqs = jax.vmap(sample_hmm_stateseq, in_axes=(0,0,0,None))(
        jr.split(key,mask.shape[0]),
        jnp.moveaxis(log_likelihoods,0,-1),
        mask.astype(float)[:,nlags:], pi)
    return stateseqs

@jax.jit
def resample_scales(key, *, x, v, h, Y, Cd, sigmasq, nu_k, **kwargs):
    Ypred = affine_transform((pad_affine(x)@Cd.T).reshape(Y.shape),v,h)
    stats = ((Ypred - Y)**2).sum(-1)
    variance = (stats/sigmasq + nu_k)/(nu_k+3)
    return sample_scaled_inv_chi2(key, nu_k+3, variance)

@jax.jit
def resample_regression_params(key, mask, x_in, x_out, nu_0, S_0, M_0, K_0):
    S_out_out = (x_out[:,:,None]*x_out[:,None,:]*mask[:,None,None]).sum(0)
    S_out_in = (x_out[:,:,None]*x_in[:,None,:]*mask[:,None,None]).sum(0)
    S_in_in = (x_in[:,:,None]*x_in[:,None,:]*mask[:,None,None]).sum(0)
    K_0_inv = inv_psd(K_0)
    K_n = inv_psd(K_0_inv + S_in_in)
    M_n = (M_0@K_0_inv + S_out_in)@K_n
    S_n = S_0 + S_out_out + (M_0@K_0_inv@M_0.T - M_n@inv_psd(K_n)@M_n.T)
    return sample_mniw(key, nu_0+mask.sum(), S_n, M_n, K_n)


@partial(jax.jit, static_argnames=('num_states','nlags'))
def resample_ar_params(key, *, nlags, num_states, mask, x, z, nu_0, S_0, M_0, K_0, **kwargs):
    x_in = pad_affine(add_lags(x, nlags)).reshape(-1,nlags*x.shape[-1]+1)
    x_out = x[...,nlags:,:].reshape(-1,x.shape[-1])
    masks = mask[...,nlags:].reshape(1,-1)*jnp.eye(num_states)[:,z.flatten()]
    return jax.vmap(resample_regression_params, in_axes=(0,0,None,None,None,None,None,None))(
        jr.split(key,num_states), masks, x_in, x_out, nu_0, S_0, M_0, K_0)


def resample_transitions(key, *, z, mask, betas, alpha, kappa, gamma, **kwargs):
    counts = jnp.array(count_transitions(len(betas), np.array(mask[...,-z.shape[-1]:]), np.array(z)))
    betas, pi = sample_hdp_transitions(key, counts, betas, alpha, kappa, gamma)
    return betas, pi


@jax.jit
def resample_latents(key, *, Y, mask, v, h, z, s, Cd, sigmasq, Ab, Q, **kwargs):
    k,nlags,n = Ab.shape[1],Ab.shape[2]//Ab.shape[1],Y.shape[0]
    ys = inverse_affine_transform(Y,v,h).reshape(*Y.shape[:-2],-1)
    A, B, Q, C, D = *ar_to_lds(Ab[...,:-1],Ab[...,-1],Q,Cd[...,:-1]),Cd[...,-1]
    R = jnp.repeat(s*sigmasq,Y.shape[-1],axis=-1)[:,nlags:]
    mu0,S0 = jnp.zeros((n,k*nlags)),jnp.repeat(jnp.eye(k*nlags)[None]*10,n,axis=0)
    xs = jax.vmap(kalman_sample, in_axes=(0,0,0,0,0,0,0,0,None,None,0))(
        jr.split(key,ys.shape[0]), ys[:,nlags:], mask[:,nlags:], mu0, S0, A[z], B[z], Q[z], C, D, R)
    return jnp.concatenate([mu0.reshape(n,nlags,k),xs[:,:,-k:]],axis=1)

@jax.jit
def resample_heading(key, *, Y, v, x, s, Cd, sigmasq, **kwargs):
    Cx_tild = (pad_affine(x)@Cd.T).reshape(Y.shape)[...,None]
    Ycen = (Y - v[...,None,:])[...,None,:]
    S = (Cx_tild*Ycen/(s*sigmasq)[...,None,None]).sum(-3)
    kappa_cos = S[...,0,0]+S[...,1,1]
    kappa_sin = S[...,0,1]-S[...,1,0]
    theta = vector_to_angle(jnp.stack([kappa_cos,kappa_sin],axis=-1))
    kappa = jnp.sqrt(kappa_cos**2 + kappa_sin**2)
    return sample_vonmises(key, theta, kappa)

    
@jax.jit 
def resample_location(key, *, mask, Y, h, x, s, Cd, sigmasq, sigmasq_loc, **kwargs):
    d = Y.shape[-1]
    gammasq = 1/(1/(s*sigmasq)).sum(-1)
    Cx_tild = (pad_affine(x)@Cd.T).reshape(Y.shape)
    rot_matrix = angle_to_rotation_matrix(-h, keypoint_dim=d)
    mu = ((Y - (rot_matrix[...,None,:,:]*Cx_tild[...,None,:]).sum(-1)) \
          *(gammasq[...,None]/(s*sigmasq))[...,None]).sum(-2)

    m0, S0 = mu[:,0], gammasq[:,0][...,None,None]*jnp.eye(d)
    As = jnp.tile(jnp.eye(d), (*mask.shape,1,1))
    Bs = jnp.zeros((*mask.shape,d))
    Qs = jnp.tile(jnp.eye(d), (*mask.shape,1,1))*sigmasq_loc
    C,D,Rs = jnp.eye(d),jnp.zeros(d),gammasq[...,None]*jnp.ones(d)
    return jax.vmap(kalman_sample, in_axes=(0,0,0,0,0,0,0,0,None,None,0))(
        jr.split(key,mask.shape[0]), mu, mask, m0, S0, As, Bs, Qs, C, D, Rs)

@jax.jit
def resample_obs_params(key, *, Y, mask, v, h, x, s, K_0, M_0, nu_0, sigmasq_0, **kwargs):
    k,d = Y.shape[-2:]
    K_0_inv = inv_psd(K_0)
    x = pad_affine(x).reshape(-1,x.shape[-1]+1)
    mask = mask.flatten()
    
    def resample_obs_k(key, y, s, M_k):
        S_xx = (x[:,:,None]*x[:,None,:]/s[:,None,None]*mask[:,None,None]).sum(0)
        S_yx = (y[:,:,None]*x[:,None,:]/s[:,None,None]*mask[:,None,None]).sum(0)
        S_yy = (y[:,:,None]*y[:,None,:]/s[:,None,None]*mask[:,None,None]).sum(0)

        nu_n = nu_0 + d*x.shape[0]
        K_n = inv_psd(K_0_inv+S_xx)
        M_n = (M_k@K_0_inv + S_yx)@K_n
        sigmasq_n = jnp.trace(
              S_yy + nu_0*sigmasq_0 
            + M_k@K_0_inv@M_k.T 
            - M_n@(K_0_inv+S_xx)@M_n.T)/nu_n
        
        sigmasq = sample_scaled_inv_chi2(key, nu_n, sigmasq_n)
        Cd = sample_mn(key, M_n, sigmasq*jnp.eye(d), K_n)
        return Cd, sigmasq
    
    Cd, sigmasq = jax.vmap(resample_obs_k, in_axes=(0,1,1,0))(
        jr.split(key,k), inverse_affine_transform(Y,v,h).reshape(-1,k,d),
        s.reshape(-1,k), M_0.reshape(k,d,x.shape[-1]))
    return jnp.concatenate(Cd,axis=0), sigmasq