""" Utility functions for going back and forth between matlab helperOC toolbox.
"""
import numpy as np
import hj_reachability as hj

def create_helperoc_grid(grid: hj.Grid):
    grid = dict()
    grid['min'] = grid.domain.lo
    grid['max'] = grid.domain.hi
    grid['N'] = grid.spacings
    grid['dim'] = len(grid.spacings)
    dx = (grid['max'] - grid['min']) / grid['N']
    grid['dx'] = dx
    vs = list()
    for i in range(grid['dim']):
        vs.append(np.linspace(grid['min'][i], grid['max'][i], N[i]))
    grid['vs'] = vs
    xs = np.meshgrid(*vs)
    grid['xs'] = xs
    # TODO: fix this by checking grid.boundary_conditions
    # grid['pdims'] = pdims
    return grid