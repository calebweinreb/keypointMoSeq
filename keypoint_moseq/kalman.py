from jax.config import config
config.update("jax_enable_x64", True)
import jax, jax.numpy as jnp, jax.random as jr
from keypoint_moseq.util import inv_psd


def _predict(m, S, A, B, Q):
    """
        Predict next mean and covariance under a linear Gaussian model
    
        p(x_{t+1}) = \int N(x_t | m, S) N(x_{t+1} | Ax_t + Bu, Q)
                    = N(x_{t+1} | Am + Bu, A S A^T + Q)
    """
    mu_pred = A @ m + B
    Sigma_pred = A @ S @ A.T + Q
    return mu_pred, Sigma_pred

def _condition_on_diag(m, S, C, D, Rdiag, y):
    """
    Condition a Gaussian potential on a new linear Gaussian observation.
    Assume R is diagonal.
    """
    Sinv = inv_psd(S)
    K = ((jnp.diag(1/Rdiag)-C@inv_psd(Sinv+(C.T/Rdiag)@C)@C.T/Rdiag/Rdiag[:,None])@C@S).T
    Sigma_cond = S - K @ C @ S
    mu_cond = Sigma_cond @ (Sinv@m + C.T @ ((y - D)/Rdiag))
    return mu_cond, Sigma_cond


def _condition_on(m, S, C, D, R, y):
    """
    Condition a Gaussian potential on a new linear Gaussian observation.
    """
    K = jnp.linalg.solve(R + C @ S @ C.T, C @ S).T
    Sigma_cond = S - K @ C @ S
    mu_cond = Sigma_cond @ (jnp.linalg.solve(S, m) + C.T @ jnp.linalg.solve(R, y - D))
    return mu_cond, Sigma_cond


def kalman_filter(ys, mask, m0, S0, As, Bs, Qs, C, D, Rs):
    """
    Run a Kalman filter to produce the marginal likelihood and filtered state 
    estimates. 

    Args:
        lds: tuple of (mu0, S0, As, bs, Qs, Cs, ds, Rs)
        ys: (T, N) array of outputs
    """

    def _step(carry, args):
        mu_pred, Sigma_pred = carry
        A, B, Q, R, y = args
        mu_cond, Sigma_cond = _condition_on_diag(mu_pred, Sigma_pred, C, D, R, y)
        mu_pred, Sigma_pred = _predict(mu_cond, Sigma_cond, A, B, Q)
        return (mu_pred, Sigma_pred), (mu_cond, Sigma_cond)
    
    def _masked_step(carry, args):
        mu_pred, Sigma_pred = carry
        return (mu_pred, Sigma_pred), (mu_pred, Sigma_pred)

    _,(filtered_mus, filtered_Sigmas) = jax.lax.scan(
        lambda carry,args: jax.lax.cond(args[0]>0, _step, _masked_step, carry, args[1:]),
        (m0, S0), (mask, As, Bs, Qs, Rs, ys))
    return filtered_mus, filtered_Sigmas

def kalman_sample(rng, ys, mask, m0, S0, As, Bs, Qs, C, D, Rs):
    """
    Run forward-filtering, backward-sampling to draw x_{1:T} | y_{1:T}. 
    """
    # Run the Kalman Filter
    filtered_mus, filtered_Sigmas = kalman_filter(ys, mask, m0, S0, As, Bs, Qs, C, D, Rs)

    # Sample backward in time
    def _step(carry, args):
        xn = carry
        rng, mu_pred, Sigma_pred, A, B, Q = args
        mu_cond, Sigma_cond = _condition_on(mu_pred, Sigma_pred, A, B, Q, xn)
        x = jr.multivariate_normal(rng, mu_cond, Sigma_cond)
        return x, x
    
    def _masked_step(carry, args):
        xn = carry
        return xn,jnp.zeros_like(xn)

    # Initialize the last state
    rng, this_rng = jr.split(rng, 2)
    xT = jr.multivariate_normal(rng,
        filtered_mus[-1], filtered_Sigmas[-1])

    args = (mask[:-1], jr.split(rng, ys.shape[0]-1), 
            filtered_mus[:-1], filtered_Sigmas[:-1],
            As[:-1],Bs[:-1],Qs[:-1])
    
    _, xs = jax.lax.scan(lambda carry,args: jax.lax.cond(
        args[0]>0, _step, _masked_step, carry, args[1:]), xT, args, reverse=True)

    return jnp.vstack([xs, xT])
