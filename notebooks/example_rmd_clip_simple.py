# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: robstat
#     language: python
#     name: robstat
# ---

# <center><strong><font size=+3>Speeding up RMD-clipping</font></center>
# <br><br>
# </center>
# <center><strong><font size=+2>Matyas Molnar and Bojan Nikolic</font><br></strong></center>
# <br><center><strong><font size=+1>Astrophysics Group, Cavendish Laboratory, University of Cambridge</font></strong></center>

# +
import os

import numpy as np
from scipy import special, stats
from sklearn.covariance import MinCovDet

from robstat.stdstat import mad_clip
from robstat.utils import DATAPATH, decomposeCArray, flt_nan
# -

vis_file = os.path.join(DATAPATH, 'sample_vis_data.npz')
vis_data = np.load(vis_file)

data = vis_data['data']
flags = np.isnan(data)
redg = vis_data['redg']
pol = vis_data['pol']
lsts = vis_data['lsts']
JDs = vis_data['JDs']
chans = vis_data['chans']
freqs = vis_data['freqs']

# parameters
sigma = 5.0  # number of normal standard deviations for clipping
min_N = 5  # minimum length of array to clip, below which no clipping is performed.

eg_data = data[:, 0, 0, 0]
points = decomposeCArray(flt_nan(eg_data.flatten()))

# +
# relate in terms of probabilities:
# the probability that a normal deviate lies in the range between  \mu - n*\sigma and \mu + n*\sigma:
chi2_p = special.erf(sigma/np.sqrt(2))
# transform this probability to chi^2 quantile
chi2_q = stats.chi2.ppf(chi2_p, df=points.shape[1])

print('χ^2 quantile corresponding to {}σ (p = {:.7f}) is {:.7f}'.\
      format(sigma, chi2_p, chi2_q))
# -

robust_cov = MinCovDet(random_state=0).fit(points)
rmd_outliers = np.where(robust_cov.mahalanobis(points) > chi2_q)[0]

# %%timeit
# RMD-clipping speed
robust_cov = MinCovDet(random_state=0).fit(points)
rmd_outliers = np.where(robust_cov.mahalanobis(points) > chi2_q)[0]

# +
# %%timeit
# MAD-clipping speed, for comparison.
# Note that MAD-clipping can be vectorized, while RMD-clipping needs to be looped (see later)
_, f_r = mad_clip(points[:, 0], sigma=sigma, min_N=min_N)
_, f_i = mad_clip(points[:, 1], sigma=sigma, min_N=min_N)

mad_outliers = np.where(f_r + f_i)[0]
# -

# get RMD ellipse parameters from covariance matrix
eig_vals, eig_vecs = np.linalg.eig(robust_cov.covariance_)
radii = np.sqrt(eig_vals)
lrg_ev = eig_vecs[np.argmax(eig_vals)]
alpha = np.arctan2(eig_vals[0] - robust_cov.covariance_[0][0], robust_cov.covariance_[0][1])



# ### Run RMD for all data

# +
import multiprocess as multiprocessing

# require a shared ctype array in order to fill in a numpy array in parallel

def create_mp_array(arr):
    shared_arr = multiprocessing.RawArray(np.ctypeslib.as_ctypes_type(arr.dtype), int(np.prod(arr.shape)))
    new_arr = np.frombuffer(shared_arr, arr.dtype).reshape(arr.shape)  # shared_arr and new_arr the same memory
    new_arr[...] = arr
    return shared_arr, new_arr

def mp_init(shared_arr_, sharred_arr_shape_, sharred_arr_dtype_):
    global shared_arr, sharred_arr_shape, sharred_arr_dtype
    shared_arr = shared_arr_
    sharred_arr_shape = sharred_arr_shape_
    sharred_arr_dtype = sharred_arr_dtype_

def mp_iter(s):
    d = data[:, s[0], s[1], s[2]]
    if not np.isnan(d).all():
        
        isfinite = np.isfinite(d).nonzero()[0]
        d = decomposeCArray(flt_nan(d))
        robust_cov = MinCovDet(random_state=0).fit(d)
        outliers = robust_cov.mahalanobis(d) > chi2_q

        rmd_clip_f = np.frombuffer(shared_arr, dtype).reshape(shape)
        rmd_clip_f[isfinite, s[0], s[1], s[2]] = outliers


# +
rmd_clip_f = np.ones_like(data, dtype=bool)
d_shared, rmd_clip_f = create_mp_array(rmd_clip_f)
dtype = rmd_clip_f.dtype
shape = rmd_clip_f.shape

m_pool = multiprocessing.Pool(multiprocessing.cpu_count(), initializer=mp_init, \
                              initargs=(d_shared, dtype, shape))
_ = m_pool.map(mp_iter, np.ndindex(data.shape[1:]))
m_pool.close()
m_pool.join()

rmd_clip_f = rmd_clip_f ^ flags

# apply min_N condition
mad_f_min_n = np.logical_not(flags).sum(axis=0) < min_N
mad_f_min_n = np.expand_dims(mad_f_min_n, axis=0)
mad_f_min_n = np.repeat(mad_f_min_n, flags.shape[0], axis=0)
rmd_clip_f[mad_f_min_n] = False

print('Number of data point flagged from RMD-clipping: {:,}'.format(rmd_clip_f.sum()))


