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
from sklearn.covariance import MinCovDet, fast_mcd, empirical_covariance, _robust_covariance

from robstat.stdstat import mad_clip
from robstat.utils import DATAPATH, decomposeCArray, flt_nan
# -

vis_file = os.path.join(DATAPATH, 'sample_vis_data.npz')
vis_data = np.load(vis_file)

data = vis_data['data']
flags = np.isnan(data)
redg = vis_data['redg']
pol = vis_data['pol']
lsts = vis_data['lsts'] # * 12 / np.pi *3600 to convert to seconds!
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
        print (d.shape)
        isfinite = np.isfinite(d).nonzero()[0]
        d = decomposeCArray(flt_nan(d))
        # print(d.shape)
        robust_cov = MinCovDet(random_state=0).fit(d)
        outliers = robust_cov.mahalanobis(d) > chi2_q

        rmd_clip_f = np.frombuffer(shared_arr, dtype).reshape(shape)
        rmd_clip_f[isfinite, s[0], s[1], s[2]] = outliers


# +

def runmult(cpus = multiprocessing.cpu_count()):
    global dtype, shape
    rmd_clip_f = np.ones_like(data, dtype=bool)
    d_shared, rmd_clip_f = create_mp_array(rmd_clip_f)
    dtype = rmd_clip_f.dtype
    shape = rmd_clip_f.shape

    if cpus:
        m_pool = multiprocessing.Pool(cpus,
                                  initializer=mp_init, \
                                  initargs=(d_shared, dtype, shape))
        _ = m_pool.map(mp_iter, np.ndindex(data.shape[1:]))
        m_pool.close()
        m_pool.join()
    else:
        mp_init(d_shared, dtype, shape)
        list(map(mp_iter, np.ndindex(data.shape[1:])))

    rmd_clip_f = rmd_clip_f ^ flags

    # apply min_N condition
    mad_f_min_n = np.logical_not(flags).sum(axis=0) < min_N
    mad_f_min_n = np.expand_dims(mad_f_min_n, axis=0)
    mad_f_min_n = np.repeat(mad_f_min_n, flags.shape[0], axis=0)
    rmd_clip_f[mad_f_min_n] = False

    print('Number of data point flagged from RMD-clipping: {:,}'.format(rmd_clip_f.sum()))



def blkSerial(data):
    "Go through the data in blocks of (frequency,time) "
    res=np.zeros_like(data)
    for ii in np.ndindex(data.shape[3:]):
        d = data[:, : , : , ii[0]]
        if np.isnan(d).all():
            continue 
        d = np.apply_along_axis( decomposeCArray, 0, d)
        rc = blkMCD(d)
        res[:, :, :, ii[0]] = rc
    return res


def blkMCD(d):
    "Begin refactoring the code. Here the fast_mcd is pulled out"
    r = MinCovDet(random_state=0)    
    ll = blk_fast_mcd(
            d,
            support_fraction=r.support_fraction,
            cov_computation_method=r._nonrobust_covariance,
            random_state=0,
        )
    res = numpy.zeros(shape=(d.shape[0], d.shape[2], d.shape[3]))
    for l in ll :
        r = MinCovDet(random_state=0)
        ii, raw_location, raw_covariance, raw_support, raw_dist = l
        r.raw_location_ = raw_location
        r.raw_covariance_ = raw_covariance
        r.raw_support_ = raw_support
        r.location_ = raw_location
        r.support_ = raw_support
        r.dist_ = raw_dist
        XX = d[:, :, ii[0], ii[1]]
        f = np.isfinite(XX).all(axis=1)
        XX = XX[f]
        r.correct_covariance(XX)
        r.reweight_covariance(XX)
        res[f, ii[0], ii[1]] = r.mahalanobis(XX) 
    return res
    
def simple_fast_mcd(X,
                    support_fraction=None,
                    cov_computation_method=empirical_covariance,
                    random_state=None,
                    ):
    res = []
    for ii in  np.ndindex(X.shape[2:]):
        XX = X[:, :, ii[0], ii[1]]
        f = np.isfinite(XX).all(axis=1)
        XX = XX[f]
        # First two axes are the main axes, others batch
        n_samples, n_features = XX.shape[0:2]
        # minimum breakdown value
        if support_fraction is None:
            n_support = int(np.ceil(0.5 * (n_samples + n_features + 1)))
        else:
            n_support = int(support_fraction * n_samples)

        
        # 1. Find the 10 best couples (location, covariance)
        # considering two iterations
        n_trials = 30
        n_best = 10
        locations_best, covariances_best, _, _ = _robust_covariance.select_candidates(
            XX,
            n_support,
            n_trials=n_trials,
            select=n_best,
            n_iter=2,
            cov_computation_method=cov_computation_method,
            random_state=random_state,
        )
        # 2. Select the best couple on the full dataset amongst the 10
        locations_full, covariances_full, supports_full, d, det = _robust_covariance.select_candidates(
            XX,
            n_support,
            n_trials=(locations_best, covariances_best),
            select=1,
            cov_computation_method=cov_computation_method,
            random_state=random_state,
        )
        location = locations_full[0]
        covariance = covariances_full[0]
        support = supports_full[0]
        dist = d[0]
        res.append(  [ii, location, covariance, support, dist] )
    return res

    

def blk_fast_mcd(X,
                    support_fraction=None,
                    cov_computation_method=empirical_covariance,
                    random_state=None,
                    ):
    """Bulk MCD calculation. First find candidates across the whole
    block (axes #1, #2) the find the best fit for each individually
    slice. 
    """
    # First candidates to start from, shared across the whole block
    locations_best, covariances_best = [], []
    for ii in  np.ndindex(X.shape[2:]):
        XX = X[:, :, ii[0], ii[1]]
        f = np.isfinite(XX).all(axis=1)
        XX = XX[f]
        # First two axes are the main axes, others batch
        n_samples, n_features = XX.shape[0:2]
        # minimum breakdown value
        if support_fraction is None:
            n_support = int(np.ceil(0.5 * (n_samples + n_features + 1)))
        else:
            n_support = int(support_fraction * n_samples)

        
        # 1. Find the 10 best couples (location, covariance)
        # considering two iterations
        n_trials = 3
        n_best = 1
        l_best, c_best, _, _ = _robust_covariance.select_candidates(
            XX,
            n_support,
            n_trials=n_trials,
            select=n_best,
            n_iter=2,
            cov_computation_method=cov_computation_method,
            random_state=random_state,
        )
        locations_best.extend(l_best)
        covariances_best.extend(c_best)

    locations_best = np.array(locations_best)
    covariances_best = np.array(covariances_best)
    res = []
    for ii in  np.ndindex(X.shape[2:]):
        XX = X[:, :, ii[0], ii[1]]
        f = np.isfinite(XX).all(axis=1)
        XX = XX[f]
        # First two axes are the main axes, others batch
        n_samples, n_features = XX.shape[0:2]
        # minimum breakdown value
        if support_fraction is None:
            n_support = int(np.ceil(0.5 * (n_samples + n_features + 1)))
        else:
            n_support = int(support_fraction * n_samples)        
        # We choose N random starting position + the starter for this
        # point from amonst the best in this block
        N = 9
        states = np.append(np.random.default_rng().permutation(len(locations_best))[:N], ii)
        locations_full, covariances_full, supports_full, d = _robust_covariance.select_candidates(
            XX,
            n_support,
            n_trials=(locations_best[states],
                      covariances_best[states]),
            select=1,
            cov_computation_method=cov_computation_method,
            random_state=random_state,
            n_iter=30
        )
        location = locations_full[0]
        covariance = covariances_full[0]
        support = supports_full[0]
        dist = d[0]
        res.append(  [ii, location, covariance, support, dist, _robust_covariance.fast_logdet(covariance)] )
    return res

    
