"""Household blocks: HA, TA, RA"""

import numpy as np
import sequence_jacobian as sj # sequence-space Jacobian toolkit
from scipy import linalg

"""Preliminaries for HA household block"""

# Income process from Kaplan Moll Violante 2018
# load KMV transition matrix, convert to discrete time, and fix rounding so rows sum to 1
Pi_e = linalg.expm(np.loadtxt('inputs/kmv_process/ymarkov_combined.txt'))
Pi_e /= np.sum(Pi_e, axis=1)[:, np.newaxis]

pi_e = sj.utilities.discretize.stationary(Pi_e)
e_grid_short = np.exp(np.loadtxt('inputs/kmv_process/ygrid_combined.txt'))
e_grid_short /= e_grid_short @ pi_e
n_e = len(pi_e)

def make_betas(beta_hi, dbeta, omega, q):
    """Return beta grid [beta_hi-dbeta, beta_high] and transition matrix,
    where q is probability of getting new random draw from [1-omega, omega]"""
    beta_lo = beta_hi - dbeta
    b_grid = np.array([beta_lo, beta_hi])
    pi_b = np.array([1 - omega, omega])
    Pi_b = (1-q)*np.eye(2) + q*np.outer(np.ones(2), pi_b)
    return b_grid, Pi_b, pi_b

@sj.het(exogenous='Pi', policy='a', backward='Va', backward_init=sj.hetblocks.hh_sim.hh_init)
def hh_raw(Va_p, a_grid, y, r, beta, eis):
    """Household block. Slightly modify sequence_jacobian.hetblocks.hh_sim.hh to allow for beta vector"""
    uc_nextgrid = beta[:, np.newaxis] * Va_p # beta now vector, multiply Va_prime by row (e state)
    c_nextgrid = uc_nextgrid ** (-eis)
    coh = (1 + r) * a_grid[np.newaxis, :] + y[:, np.newaxis]
    a = sj.interpolate.interpolate_y(c_nextgrid + a_grid, coh, a_grid)
    sj.misc.setmin(a, a_grid[0])
    c = coh - a 
    Va = (1 + r) * c ** (-1 / eis)
    return Va, a, c

def make_grids(min_a, max_a, n_a, beta_hi, dbeta, omega, q):
    # asset and beta grids
    a_grid = sj.grids.asset_grid(min_a, max_a, n_a)
    b_grid_short, Pi_b, pi_b = make_betas(beta_hi, dbeta, omega, q)

    # combine grids for beta and e (latter pre-loaded from KMV above)
    e_grid = np.kron(np.ones_like(b_grid_short), e_grid_short)
    beta = np.kron(b_grid_short, np.ones_like(e_grid_short))
    Pi = np.kron(Pi_b, Pi_e)
    pi_pdf = np.kron(pi_b, pi_e)
        
    return e_grid, Pi, a_grid, beta, pi_pdf

def income(wN_aftertax, N, e_grid, Tr_lumpsum, Tax_richest, zeta, pi_pdf):
    # Auclert-Rognlie 2020 incidence function for labor income, with cyclicality parameter zeta
    # in default case with zeta = 0, this is just gamma / N = 1 and irrelevant
    gamma_N = e_grid ** (zeta * np.log(N)) / np.vdot(e_grid ** (zeta * np.log(N)), pi_pdf)
    
    # net after-tax income (include lump-sum transfer option)
    y = wN_aftertax * e_grid * gamma_N + Tr_lumpsum

    # also add option to tax richest type at margin
    y = y.reshape(-1, n_e)                   # reshape to beta*e grid
    y[:, -1] -= Tax_richest / pi_e[-1].sum() # tax richest e type
    y = y.ravel()                            # flatten back
    return y


"""Consolidated HA household block and calibration of all exogenously-set parameters"""

hh_ha = hh_raw.add_hetinputs([make_grids, income])
hh_ha.name = 'hh_ha'
calibration_ha = dict(eis=1, min_a=0, max_a = 4000, n_a=200, r=0.005, q=0.01, Tr_lumpsum=0, Tax_richest=0, zeta=0)

# note that after-tax wages and labor are determined in equilibrium in full model, but we'll calibrate hh separately first
# steady-state normalized to Y = N = 1, out of which asset income on A = 20 at r = 0.005 is 0.1,
# and government spending is G=0.2, so markups + taxes total take 0.3
calibration_ha['N'], calibration_ha['wN_aftertax'] = 1, 0.7


"""Two-agent and rep-agent blocks"""

@sj.solved(unknowns={'C_RA': 1, 'A': 1},
           targets=["euler", "budget_constraint"])
def hh_ta(C_RA, A, wN_aftertax, eis, beta, r, lam):
    euler = (beta * (1 + r(+1))) ** (-eis) * C_RA(+1) - C_RA    # Euler eq for consumption of infinitely lived household
    C_H2M = wN_aftertax                                         # consumption of hand to mouth household
    C = (1 - lam) * C_RA + lam * C_H2M
    budget_constraint = (1 + r) * A(-1) + wN_aftertax  - C - A  # budget constraint for infinitely lived household
    return euler, budget_constraint, C_H2M, C


@sj.solved(unknowns={'C': 1, 'A': 1},
           targets=["euler", "budget_constraint"])
def hh_ra(C, A, wN_aftertax, eis, beta, r):
    euler = (beta * (1 + r(+1)))**(-eis) * C(+1) - C
    budget_constraint = (1 + r) * A(-1) + wN_aftertax - C - A
    return euler, budget_constraint
