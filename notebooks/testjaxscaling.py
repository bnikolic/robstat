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

import numpy
import jax.numpy

N=30
matrix = numpy.matrix(numpy.random.rand(N, N))
matrix = matrix + matrix.getH()


#Basic:
# %timeit jax.numpy.linalg.pinv(matrix).block_until_ready()
# %timeit jax.numpy.linalg.pinv(numpy.array([matrix,]*10)).block_until_ready()
