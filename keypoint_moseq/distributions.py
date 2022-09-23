from jax.config import config
import jax, jax.numpy as jnp, jax.random as jr
import tensorflow_probability.substrates.jax.distributions as tfd

def sample_vonmises(key, theta, kappa):
    return tfd.VonMises(theta, kappa).sample(seed=key)

def sample_gamma(key, a, b):
    return jr.gamma(key, a) / b

def sample_inv_gamma(key, a, b):
    return 1/sample_gamma(key, a, b)

def sample_scaled_inv_chi2(key, degs, variance):
    return sample_inv_gamma(key, degs/2, degs*variance/2)

def sample_chi2(key, degs):
    return jr.gamma(key, degs/2)*2

def sample_discrete(key, distn,dtype=jnp.int32):
    return jr.categorical(key, jnp.log(distn))

def sample_mn(key, M, U, V):
    G = jr.normal(key,M.shape)
    G = jnp.dot(jnp.linalg.cholesky(U),G)
    G = jnp.dot(G,jnp.linalg.cholesky(V).T)
    return M + G

def sample_invwishart(key,S,nu):
    n = S.shape[0]
    chol = jnp.linalg.cholesky(S)
    chi2_key, norm_key = jr.split(key)
    x = jnp.diag(jnp.sqrt(sample_chi2(chi2_key, nu-jnp.arange(n))))
    x = x.at[jnp.triu_indices_from(x,1)].set(jr.normal(norm_key, (n*(n-1)//2,)))
    R = jnp.linalg.qr(x,'r')
    T = jax.scipy.linalg.solve_triangular(R.T,chol.T,lower=True).T
    return jnp.dot(T,T.T)

def sample_mniw(key, nu, S, M, K):
    sigma = sample_invwishart(key, S, nu)
    A = sample_mn(key, M, sigma, K)
    return A, sigma



def sample_hmm_stateseq(key, log_likelihoods, mask, pi):
    """
    Use the forward-backward algorithm to sample state-sequences in a Markov chain.
    
    """
    def _forward_message(carry, args):
        ll_t, mask_t = args
        in_potential, logtot = carry
        cmax = ll_t.max()
        alphan_t = in_potential * jnp.exp(ll_t - cmax)
        norm = alphan_t.sum() + 1e-16
        alphan_t = alphan_t / norm
        logprob = jnp.log(norm) + cmax
        in_potential = alphan_t.dot(pi)*mask_t + in_potential*(1-mask_t)
        return (in_potential, logtot + logprob*mask_t), alphan_t    

    def _sample(args):
        key, next_potential, alphan_t = args
        key, newkey = jr.split(key)
        s = sample_discrete(newkey, next_potential * alphan_t)
        next_potential = pi[:,s]
        return (key,next_potential), s

    def _backward_message(carry, args):
        key, next_potential = carry
        alphan_t, mask_t = args
        return jax.lax.cond(mask_t>0, _sample, lambda args: (args[:-1],0), (key, next_potential, alphan_t))
        
    init_distn = jnp.ones(pi.shape[0])/pi.shape[0]
    (_,log_likelihood), alphan = jax.lax.scan(_forward_message,  (init_distn,0.), (log_likelihoods, mask))
    
    init_potential = jnp.ones(pi.shape[0])
    _,stateseq = jax.lax.scan(_backward_message, (key,init_potential), (alphan,mask), reverse=True)
    return stateseq, log_likelihood


