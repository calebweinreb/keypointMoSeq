# Keypoint MoSeq

Implementation of [a keypoint-adapted-SLDS](https://www.overleaf.com/read/rkpnxchbdrrb) aproach for unsupervised behavior discovery. Check out the [example keypoint notebook](examples/keypoint_slds.ipynb) for how to use the code.


The standard MoSeq HDP-ARHMM can also be run using code from this repo. The model is identical to that in [pyhsmm-autoregressive](https://github.com/mattjj/pyhsmm-autoregressive). The key differences are that this new code is written with jax; lacks C-dependencies; and is functional rather than object oriented. Example AR-HMM code is provided in the [depth-PCA notebook](examples/depth_ar_hmm.ipynb). 
