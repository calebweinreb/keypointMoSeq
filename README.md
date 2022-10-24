# Keypoint MoSeq

Implementation of [a keypoint-adapted-SLDS](https://www.overleaf.com/read/rkpnxchbdrrb) aproach for unsupervised behavior discovery. Check out the [example keypoint notebook](examples/keypoint_slds.ipynb) for how to use the code.


The standard MoSeq HDP-ARHMM can also be run using code from this repo. The model is identical to that in [pyhsmm-autoregressive](https://github.com/mattjj/pyhsmm-autoregressive). The key differences are that this new code is written with jax; lacks C-dependencies; and is functional rather than object oriented. Example AR-HMM code is provided in the [depth-PCA notebook](examples/depth_ar_hmm.ipynb). 


## Installation

Install [Anaconda](https://docs.anaconda.com/anaconda/install/index.html) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html). Create and activate an environment called `keypoint_moseq` with python≥3.7:
```
conda create -n keypoint_moseq python=3.9
conda activate keypoint_moseq
```

Install opencv and jax
```
conda install -c conda-forge opencv
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Clone or download the this github repo and pip install:
```
git clone https://github.com/calebweinreb/keypointMoSeq.git
pip install -e keypoint_moseq
```

Make the new environment accessible in jupyter 
```
python -m ipykernel install --user --name=keypoint_moseq
```