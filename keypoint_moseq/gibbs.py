from jax.config import config
config.update('jax_enable_x64', True)
import jax, jax.numpy as jnp, jax.random as jr
from keypoint_moseq.util import *
from keypoint_moseq.distributions import *
from keypoint_moseq.transitions import sample_hdp_transitions, sample_transitions
from keypoint_moseq.kalman import kalman_sample
na = jnp.newaxis

@jax.jit
def resample_latents(key, *, Y, mask, v, h, z, s, Cd, sigmasq, Ab, Q, **kwargs):
    d,nlags,n = Ab.shape[1],Ab.shape[2]//Ab.shape[1],Y.shape[0]
    Gamma = center_embedding(Y.shape[-2])
    Cd = jnp.kron(Gamma, jnp.eye(Y.shape[-1])) @ Cd
    ys = inverse_affine_transform(Y,v,h).reshape(*Y.shape[:-2],-1)
    A, B, Q, C, D = *ar_to_lds(Ab[...,:-1],Ab[...,-1],Q,Cd[...,:-1]),Cd[...,-1]
    R = jnp.repeat(s*sigmasq,Y.shape[-1],axis=-1)[:,nlags:]
    mu0,S0 = jnp.zeros((n,d*nlags)),jnp.repeat(jnp.eye(d*nlags)[na]*10,n,axis=0)
    xs = jax.vmap(kalman_sample, in_axes=(0,0,0,0,0,0,0,0,0,na,na,0))(
        jr.split(key, ys.shape[0]), ys[:,nlags:], mask[:,nlags:], 
        mu0, S0, A[z], B[z], Q[z], jnp.linalg.inv(Q)[z], C, D, R)
    xs = jnp.concatenate([xs[:,0,:-d].reshape(-1,nlags-1,d)[::-1], xs[:,:,-d:]],axis=1)
    return xs

@jax.jit
def resample_heading(key, *, Y, v, x, s, Cd, sigmasq, **kwargs):
    k,d = Y.shape[-2:]
    Gamma = center_embedding(k)
    Ybar = Gamma @ (pad_affine(x)@Cd.T).reshape(*Y.shape[:-2],k-1,d)
    S = (Ybar[...,na]*(Y - v[...,na,:])[...,na,:]/(s*sigmasq)[...,na,na]).sum(-3)
    kappa_cos = S[...,0,0]+S[...,1,1]
    kappa_sin = S[...,0,1]-S[...,1,0]
    theta = vector_to_angle(jnp.stack([kappa_cos,kappa_sin],axis=-1))
    kappa = jnp.sqrt(kappa_cos**2 + kappa_sin**2)
    return sample_vonmises(key, theta, kappa)

    
@jax.jit 
def resample_location(key, *, mask, Y, h, x, s, Cd, sigmasq, sigmasq_loc, **kwargs):
    k,d = Y.shape[-2:]
    Gamma = center_embedding(k)
    gammasq = 1/(1/(s*sigmasq)).sum(-1)
    Ybar = Gamma @ (pad_affine(x)@Cd.T).reshape(*Y.shape[:-2],k-1,d)
    rot_matrix = angle_to_rotation_matrix(h, keypoint_dim=d)
    mu = ((Y - (rot_matrix[...,na,:,:]*Ybar[...,na,:]).sum(-1)) \
          *(gammasq[...,na]/(s*sigmasq))[...,na]).sum(-2)

    m0, S0 = mu[:,0], gammasq[:,0][...,na,na]*jnp.eye(d)
    As = jnp.tile(jnp.eye(d), (*mask.shape,1,1))
    Bs = jnp.zeros((*mask.shape,d))
    Qs = jnp.tile(jnp.eye(d), (*mask.shape,1,1))*sigmasq_loc
    Qinvs = jnp.tile(jnp.eye(d), (*mask.shape,1,1))/sigmasq_loc
    C,D,Rs = jnp.eye(d),jnp.zeros(d),gammasq[...,na]*jnp.ones(d)
    return jax.vmap(kalman_sample, in_axes=(0,0,0,0,0,0,0,0,0,na,na,0))(
        jr.split(key,mask.shape[0]), mu, mask, m0, S0, As, Bs, Qs, Qinvs, C, D, Rs)[...,:-1,:]



@jax.jit
def resample_obs_params(key, *, Y, mask, sigmasq, v, h, x, s, sigmasq_C, **kwargs):
    k,d,D = *Y.shape[-2:],x.shape[-1]
    Gamma = center_embedding(k)
    mask = mask.flatten()
    s = s.reshape(-1,k)
    x = x.reshape(-1,D)
    xt = pad_affine(x)

    Sinv = jnp.eye(k)[na,:,:]/s[:,:,na]/sigmasq[na,:,na]
    xx_flat = (xt[:,:,na]*xt[:,na,:]).reshape(xt.shape[0],-1).T
    # serialize this step because of memory constraints
    mGSG = mask[:,na,na] * Gamma.T@Sinv@Gamma
    S_xx_flat = jax.lax.map(lambda xx_ij: (xx_ij[:,na,na]*mGSG).sum(0), xx_flat)
    S_xx = S_xx_flat.reshape(D+1,D+1,k-1,k-1)
    S_xx = jnp.kron(jnp.concatenate(jnp.concatenate(S_xx,axis=-2),axis=-1),jnp.eye(d))
    Sigma_n = jnp.linalg.inv(jnp.eye(d*(D+1)*(k-1))/sigmasq_C + S_xx)

    vecY = inverse_affine_transform(Y, v, h).reshape(-1,k*d)
    S_yx = (mask[:,na,na]*vecY[:,:,na]*jnp.kron(
        jax.vmap(jnp.kron)(xt[:,na,:],Sinv@Gamma), 
        jnp.eye(d))).sum((0,1))         
    mu_n = Sigma_n@S_yx
                         
    return jr.multivariate_normal(key, mu_n, Sigma_n).reshape(D+1,d*(k-1)).T

@jax.jit
def resample_obs_variance(key, *, Y, mask, Cd, v, h, x, s, nu_sigma, sigmasq_0, **kwargs):
    k,d = Y.shape[-2:]
    s = s.reshape(-1,k)
    mask = mask.flatten()
    x = x.reshape(-1,x.shape[-1])
    Gamma = center_embedding(k)
    Ybar = Gamma @ (pad_affine(x)@Cd.T).reshape(-1,k-1,d)
    Ytild = inverse_affine_transform(Y,v,h).reshape(-1,k,d)
    S_y = (mask[:,na]*((Ytild - Ybar)**2).sum(-1)/s).sum(0)
    variance = (nu_sigma*sigmasq_0 + S_y)/(nu_sigma+3*mask.sum())
    degs = (nu_sigma+3*mask.sum())*jnp.ones_like(variance)
    return sample_scaled_inv_chi2(key, degs, variance)



def _ar_log_likelihood(x, params):
    Ab, Q = params
    nlags = Ab.shape[-1]//Ab.shape[-2]
    mu = pad_affine(get_lags(x, nlags))@Ab.T
    return tfd.MultivariateNormalFullCovariance(mu, Q).log_prob(x[...,nlags:,:])


@jax.jit
def resample_stateseqs(key, *, x, mask, Ab, Q, pi, **kwargs):
    nlags = Ab.shape[2]//Ab.shape[1]
    log_likelihoods = jax.lax.map(partial(_ar_log_likelihood,x), (Ab, Q))
    stateseqs = jax.vmap(sample_hmm_stateseq, in_axes=(0,0,0,na))(
        jr.split(key,mask.shape[0]),
        jnp.moveaxis(log_likelihoods,0,-1),
        mask.astype(float)[:,nlags:], pi)
    return stateseqs

@jax.jit
def resample_scales(key, *, x, v, h, Y, Cd, sigmasq, nu_s, s_0, **kwargs):
    k,d = Y.shape[-2:]
    Gamma = center_embedding(k)
    Ybar = Gamma @ (pad_affine(x)@Cd.T).reshape(*Y.shape[:-2],k-1,d)
    Ytild = inverse_affine_transform(Y,v,h)
    variance = (((Ytild - Ybar)**2).sum(-1)/sigmasq + s_0*nu_s)/(nu_s+3)
    degs = (nu_s+3)*jnp.ones_like(variance)
    return sample_scaled_inv_chi2(key, degs, variance)

@jax.jit
def resample_regression_params(key, mask, x_in, x_out, nu_0, S_0, M_0, K_0):
    S_out_out = (x_out[:,:,na]*x_out[:,na,:]*mask[:,na,na]).sum(0)
    S_out_in = (x_out[:,:,na]*x_in[:,na,:]*mask[:,na,na]).sum(0)
    S_in_in = (x_in[:,:,na]*x_in[:,na,:]*mask[:,na,na]).sum(0)
    K_0_inv = jnp.linalg.inv(K_0)
    K_n = jnp.linalg.inv(K_0_inv + S_in_in)
    M_n = (M_0@K_0_inv + S_out_in)@K_n
    S_n = S_0 + S_out_out + (M_0@K_0_inv@M_0.T - M_n@jnp.linalg.inv(K_n)@M_n.T)
    return sample_mniw(key, nu_0+mask.sum(), S_n, M_n, K_n)


@partial(jax.jit, static_argnames=('num_states','nlags'))
def resample_ar_params(key, *, nlags, num_states, mask, x, z, nu_0, S_0, M_0, K_0, **kwargs):
    x_in = pad_affine(get_lags(x, nlags)).reshape(-1,nlags*x.shape[-1]+1)
    x_out = x[...,nlags:,:].reshape(-1,x.shape[-1])
    masks = mask[...,nlags:].reshape(1,-1)*jnp.eye(num_states)[:,z.flatten()]
    return jax.vmap(resample_regression_params, in_axes=(0,0,na,na,na,na,na,na))(
        jr.split(key,num_states), masks, x_in, x_out, nu_0, S_0, M_0, K_0)


def resample_hdp_transitions(key, *, z, mask, betas, alpha, kappa, gamma, num_states, **kwargs):
    counts = jnp.array(count_transitions(num_states, np.array(mask[...,-z.shape[-1]:]), np.array(z)))
    betas, pi = sample_hdp_transitions(key, counts, betas, alpha, kappa, gamma)
    return betas, pi

def resample_transitions(key, *, z, mask, alpha, kappa, num_states, **kwargs):
    counts = jnp.array(count_transitions(num_states, np.array(mask[...,-z.shape[-1]:]), np.array(z)))
    pi = sample_transitions(key, counts, alpha, kappa)
    return pi

