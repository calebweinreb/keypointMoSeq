from jax.config import config
config.update('jax_enable_x64', True)
import jax, jax.numpy as jnp, jax.random as jr
from keypoint_moseq.util import ensure_symmetric
na = jnp.newaxis




def kalman_filter(ys, mask, m0, S0, As, Bs, Qs, C, D, Rs):
    """
    Run a Kalman filter to produce the marginal likelihood and filtered state 
    estimates. 
    """
    def _step(carry, args):
        m_pred, S_pred = carry
        A, B, Q, CRC, CRyD = args
        # condition
        S_pred_inv = jnp.linalg.inv(S_pred)
        S_cond = jnp.linalg.inv(S_pred_inv + CRC)
        m_cond = S_cond @ (S_pred_inv @ m_pred + CRyD)
        # predict
        m_pred = A @ m_cond + B
        S_pred = ensure_symmetric(A @ S_cond @ A.T + Q)
        S_pred = S_pred + jnp.eye(S_pred.shape[0])*1e-4
        return (m_pred, S_pred), (m_cond, S_cond)
    
    def _masked_step(carry, args):
        m_pred, S_pred = carry
        return (m_pred, S_pred), (m_pred, S_pred)

    CRCs = C.T@(C/Rs[...,na])
    CRyDs = ((ys-D)/Rs)@C
    
    _,(filtered_mus, filtered_Sigmas) = jax.lax.scan(
        lambda carry,args: jax.lax.cond(args[0]>0, _step, _masked_step, carry, args[1:]),
        (m0, S0), (mask, As, Bs, Qs, CRCs, CRyDs))
    return filtered_mus, filtered_Sigmas



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
    Ss = jax.lax.map(jnp.linalg.inv, filtered_Sinvs + AQinvs@As)
    means = (Ss @ filtered_Sinvs @ filtered_ms[...,na])[...,0]
    samples = jr.multivariate_normal(rng, means, Ss)
    SAQinvs = Ss @ AQinvs

    # initialize the last state
    x = jr.multivariate_normal(rng, filtered_ms[-1], filtered_Ss[-1])
    
    # scan (reverse direction)
    args = (mask, SAQinvs, Bs, samples)
    _, xs = jax.lax.scan(lambda carry,args: jax.lax.cond(
        args[0]>0, _step, _masked_step, carry, args[1:]), x, args, reverse=True)
    return jnp.vstack([xs, x])


