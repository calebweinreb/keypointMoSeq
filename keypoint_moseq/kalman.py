from jax.config import config
config.update('jax_enable_x64', True)
import jax, jax.numpy as jnp, jax.random as jr
from keypoint_moseq.util import ensure_symmetric
na = jnp.newaxis



'''
def kalman_filter(ys, mask, m0, S0, As, Bs, Qs, C, D, Rs):
    """
    Run a Kalman filter to produce the marginal likelihood and filtered state 
    estimates. 
    """
    
    def _cond(m_pred, S_pred, CRC, CRyD):
        S_pred_inv = jnp.linalg.inv(S_pred)
        S_cond = jnp.linalg.inv(S_pred_inv + CRC)
        m_cond = S_cond @ (S_pred_inv @ m_pred + CRyD)
        return m_cond, S_cond
        
    def _step(carry, args):
        m_pred, S_pred = carry
        A, B, Q, CRC, CRyD = args
        m_cond, S_cond = _cond(m_pred, S_pred, CRC, CRyD)
        m_pred = A @ m_cond + B
        S_pred = ensure_symmetric(A @ S_cond @ A.T + Q)
        S_pred = S_pred + jnp.eye(S_pred.shape[0])*1e-4
        return (m_pred, S_pred), (m_cond, S_cond)
    
    def _masked_step(carry, args):
        m_pred, S_pred = carry
        return (m_pred, S_pred), (m_pred, S_pred)

    CRCs = C.T@(C/Rs[...,na])
    CRyDs = ((ys-D)/Rs)@C
    
    (m_pred, S_pred),(filtered_ms, filtered_Ss) = jax.lax.scan(
        lambda carry,args: jax.lax.cond(args[0]>0, _step, _masked_step, carry, args[1:]),
        (m0, S0), (mask, As, Bs, Qs, CRCs[:-1], CRyDs[:-1]))
    m_cond, S_cond = _cond(m_pred, S_pred, CRCs[-1], CRyDs[-1])
    filtered_ms = jnp.concatenate((filtered_ms,m_cond[na]),axis=0)
    filtered_Ss = jnp.concatenate((filtered_Ss,S_cond[na]),axis=0)
    return filtered_ms, filtered_Ss



def kalman_sample(rng, ys, mask, m0, S0, As, Bs, Qs, Qinvs, C, D, Rs):
    # run the kalman filter
    filtered_ms, filtered_Ss = kalman_filter(ys, mask, m0, S0, As, Bs, Qs, C, D, Rs)

    def _step(x, args):
        SAQinv, B, sample = args
        x = SAQinv @ (x-B) + sample
        return x, x
    
    def _masked_step(x, args):
        return x,jnp.zeros_like(x)
    
    # precompute and sample
    AQinvs = jnp.swapaxes(As,-2,-1)@Qinvs
    filtered_Sinvs = jax.lax.map(jnp.linalg.inv, filtered_Ss)
    Ss = jax.lax.map(jnp.linalg.inv, filtered_Sinvs[:-1] + AQinvs@As)
    means = (Ss @ filtered_Sinvs[:-1] @ filtered_ms[:-1,...,na])[...,0]
    samples = jr.multivariate_normal(rng, means, Ss)
    SAQinvs = Ss @ AQinvs

    # initialize the last state
    x = jr.multivariate_normal(rng, filtered_ms[-1], filtered_Ss[-1])
    
    # scan (reverse direction)
    args = (mask, SAQinvs, Bs, samples)
    _, xs = jax.lax.scan(lambda carry,args: jax.lax.cond(
        args[0]>0, _step, _masked_step, carry, args[1:]), x, args, reverse=True)
    return jnp.vstack([xs, x])
'''



def kalman_filter(ys, mask, zs, m0, S0, A, B, Q, C, D, Rs):
    """
    Run a Kalman filter to produce the marginal likelihood and filtered state 
    estimates. 
    """

    def _predict(m, S, A, B, Q):
        mu_pred = A @ m + B
        Sigma_pred = A @ S @ A.T + Q
        return mu_pred, Sigma_pred

    def _condition_on(m, S, C, D, R, y):
        Sinv = jnp.linalg.inv(S)
        S_cond = jnp.linalg.inv(Sinv + (C.T / R) @ C)
        m_cond = S_cond @ (Sinv @ m + (C.T / R) @ (y-D))
        return m_cond, S_cond
    
    def _step(carry, args):
        m_pred, S_pred = carry
        z, y, R = args

        m_cond, S_cond = _condition_on(
            m_pred, S_pred, C, D, R, y)
        
        m_pred, S_pred = _predict(
            m_cond, S_cond, A[z], B[z], Q[z])
        
        return (m_pred, S_pred), (m_cond, S_cond)
    
    def _masked_step(carry, args):
        m_pred, S_pred = carry
        return (m_pred, S_pred), (m_pred, S_pred)
    
    
    (m_pred, S_pred),(filtered_ms, filtered_Ss) = jax.lax.scan(
        lambda carry,args: jax.lax.cond(args[0]>0, _step, _masked_step, carry, args[1:]),
        (m0, S0), (mask, zs, ys[:-1], Rs[:-1]))
    m_cond, S_cond = _condition_on(m_pred, S_pred, C, D, Rs[-1], ys[-1])
    filtered_ms = jnp.concatenate((filtered_ms,m_cond[na]),axis=0)
    filtered_Ss = jnp.concatenate((filtered_Ss,S_cond[na]),axis=0)
    return filtered_ms, filtered_Ss


@jax.jit
def kalman_sample(rng, ys, mask, zs, m0, S0, A, B, Q, C, D, Rs):
    
    # run the kalman filter
    filtered_ms, filtered_Ss = kalman_filter(ys, mask, zs, m0, S0, A, B, Q, C, D, Rs)
    
    def _condition_on(m, S, A, B, Qinv, x):
        Sinv = jnp.linalg.inv(S)
        S_cond = jnp.linalg.inv(Sinv + A.T @ Qinv @ A)
        m_cond = S_cond @ (Sinv @ m + A.T @ Qinv @ (x-B))
        return m_cond, S_cond

    def _step(x, args):
        m_pred, S_pred, z, w = args
        m_cond, S_cond = _condition_on(m_pred, S_pred, A[z], B[z], Qinv[z], x)
        L = jnp.linalg.cholesky(S_cond)
        x = L @ w + m_cond
        return x, x
    
    def _masked_step(x, args):
        return x,jnp.zeros_like(x)
    
    # precompute and sample
    Qinv = jnp.linalg.inv(Q)
    samples = jr.normal(rng, filtered_ms[:-1].shape)

    # initialize the last state
    x = jr.multivariate_normal(rng, filtered_ms[-1], filtered_Ss[-1])
    
    # scan (reverse direction)
    args = (mask, filtered_ms[:-1], filtered_Ss[:-1], zs, samples)
    _, xs = jax.lax.scan(lambda carry,args: jax.lax.cond(
        args[0]>0, _step, _masked_step, carry, args[1:]), x, args, reverse=True)
    return jnp.vstack([xs, x])

