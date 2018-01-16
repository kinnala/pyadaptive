import numpy as np
from oct2py import octave

def bisect3(p, t, marked):
    """
    Wrapper to iFEM.
    """
    import copy
    octave.addpath('pyadaptive')
    P, T = octave.bisect3(p.T, t.T+1, marked+1, nout=2)
    return P.T, T.T.astype(np.int64)-1
