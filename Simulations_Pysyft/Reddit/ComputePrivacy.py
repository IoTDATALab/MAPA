import math
import mpmath as mp
import numpy as np
import sys
#######计算隐私大小，sample_ratio:抽样比例，variance:噪声方差/ClipBound，##############
# iterations:迭代次数，delta:损失概率，max_order:lambda范围############################
def ComputePrivacy(sample_ration, variance, iterataions, delta, max_order):
    log_moments = []
    for order in range(1, max_order + 1):
        log_moment = compute_log_moment(sample_ration, variance, iterataions, order)
        log_moments.append((order, log_moment))
    min_epsilon = _compute_eps(log_moments, delta)
    return min_epsilon
###########################计算MA里的T*alpha_M^(lambda)################################
def compute_log_moment(sample_ration, variance, iterataions, order):
    mu0 = lambda y: pdf_gauss_mp(y, sigma=variance, mean=mp.mpf(0))
    mu1 = lambda y: pdf_gauss_mp(y, sigma=variance, mean=mp.mpf(1))
    mu = lambda y: (1 - sample_ration) * mu0(y) + sample_ration * mu1(y)
    a_lambda_fn = lambda z: mu(z) * (mu(z) / mu0(z)) ** order
    integral, _ = mp.quad(a_lambda_fn, [-mp.inf, mp.inf], error=True)
    moment = _to_np_float64(integral)
    return np.log(moment)*iterataions
###############################定义Gaussian密度函数######################################
def pdf_gauss_mp(x, sigma, mean):
  return mp.mpf(1.) / mp.sqrt(mp.mpf("2.") * sigma ** 2 * mp.pi) * mp.exp(
      - (x - mean) ** 2 / (mp.mpf("2.") * sigma ** 2))
##############################转化为64位浮点型###########################################
def _to_np_float64(v):
  if math.isnan(v) or math.isinf(v):
    return np.inf
  return np.float64(v)
##############################求epsilon最小值############################################
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
