from scipy.spatial.distance import *

def kl_divergence(p_probs, q_probs):
    """"KL (p || q)"""
    kl_div = p_probs * np.log(p_probs / q_probs)
    return np.sum(kl_div[np.isfinite(kl_div)])