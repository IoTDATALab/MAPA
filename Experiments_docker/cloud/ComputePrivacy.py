#!/usr/bin/python
# -*- coding: UTF-8 -*-

import math
import mpmath as mp
import numpy as np
import sys

def ComputePrivacy(sample_ration, variance, iterataions, delta, max_order):
    log_moments = []
    for order in range(1, max_order + 1):
        log_moment = compute_log_moment(sample_ration, variance, iterataions, order)
        log_moments.append((order, log_moment))
    min_epsilon = _compute_eps(log_moments, delta)
    return min_epsilon

def compute_log_moment(sample_ration, variance, iterataions, order):
    mu0 = lambda y: pdf_gauss_mp(y, sigma=variance, mean=mp.mpf(0))
    mu1 = lambda y: pdf_gauss_mp(y, sigma=variance, mean=mp.mpf(1))
    mu = lambda y: (1 - sample_ration) * mu0(y) + sample_ration * mu1(y)
    a_lambda_fn = lambda z: mu(z) * (mu(z) / mu0(z)) ** order
    integral, _ = mp.quad(a_lambda_fn, [-mp.inf, mp.inf], error=True)
    moment = _to_np_float64(integral)
    return np.log(moment)*iterataions

def pdf_gauss_mp(x, sigma, mean):
  return mp.mpf(1.) / mp.sqrt(mp.mpf("2.") * sigma ** 2 * mp.pi) * mp.exp(
      - (x - mean) ** 2 / (mp.mpf("2.") * sigma ** 2))

def _to_np_float64(v):
  if math.isnan(v) or math.isinf(v):
    return np.inf
  return np.float64(v)

def _compute_eps(log_moments, delta):
    min_eps = float("inf")
    for moment_order, log_moment in log_moments:
        # if moment_order == 0:
        #     continue
        # if math.isinf(log_moment) or math.isnan(log_moment):
        #     sys.stderr.write("The %d-th order is inf or Nan\n" % moment_order)
        #     continue
        min_eps = min(min_eps, (log_moment - math.log(delta)) / moment_order)
    return min_eps
