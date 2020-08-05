"""A package for conforming adaptive refinement of simplical meshes."""

import numpy as np
from numpy import ndarray


__version__ = '0.1.0'


def _sort_tri(p, t):
    """Make (0, 2) the longest edge in t."""

    l01 = np.sqrt(np.sum((p[:, t[0]] - p[:, t[1]]) ** 2, axis=0))
    l12 = np.sqrt(np.sum((p[:, t[1]] - p[:, t[2]]) ** 2, axis=0))
    l02 = np.sqrt(np.sum((p[:, t[0]] - p[:, t[2]]) ** 2, axis=0))

    ix01 = (l01 > l02) * (l01 > l12)
    ix12 = (l12 > l01) * (l12 > l02)

    # row swaps
    tmp = t[2, ix01]
    t[2, ix01] = t[1, ix01]
    t[1, ix01] = tmp

    tmp = t[0, ix12]
    t[0, ix12] = t[1, ix12]
    t[1, ix12] = tmp

    return t


def _find_facets(facets, t2f, marked_elems):
    """Find the facets to split."""

    split_facets = np.zeros(facets.shape[1], dtype=np.int64)
    split_facets[t2f[:, marked_elems].flatten('F')] = 1
    prev_nnz = -1e10

    while np.count_nonzero(split_facets) - prev_nnz > 0:
        prev_nnz = np.count_nonzero(split_facets)
        t2facets = split_facets[t2f]
        t2facets[2, t2facets[0] + t2facets[1] > 0] = 1
        split_facets[t2f[t2facets == 1]] = 1

    return split_facets


def _split_tris(p, t, facets, t2f, split_facets):
    """Define new triangles."""

    ix = (-1) * np.ones(facets.shape[1], dtype=np.int64)
    ix[split_facets == 1] = (np.arange(np.count_nonzero(split_facets))
                             + p.shape[1])
    ix = ix[t2f]

    red =   (ix[0] >= 0) * (ix[1] >= 0) * (ix[2] >= 0)  # noqa
    blue1 = (ix[0] ==-1) * (ix[1] >= 0) * (ix[2] >= 0)  # noqa
    blue2 = (ix[0] >= 0) * (ix[1] ==-1) * (ix[2] >= 0)  # noqa
    green = (ix[0] ==-1) * (ix[1] ==-1) * (ix[2] >= 0)  # noqa
    rest =  (ix[0] ==-1) * (ix[1] ==-1) * (ix[2] ==-1)  # noqa

    # new red elements
    t_red = np.hstack((
        np.vstack(( t[0, red], ix[0, red], ix[2, red])),  # noqa
        np.vstack(( t[1, red], ix[0, red], ix[1, red])),  # noqa
        np.vstack(( t[2, red], ix[1, red], ix[2, red])),  # noqa
        np.vstack((ix[1, red], ix[2, red], ix[0, red])),  # noqa
    ))

    # new blue elements
    t_blue1 = np.hstack((
        np.vstack((t[1, blue1],  t[0, blue1], ix[2, blue1])),  # noqa
        np.vstack((t[1, blue1], ix[1, blue1], ix[2, blue1])),  # noqa
        np.vstack((t[2, blue1], ix[2, blue1], ix[1, blue1])),  # noqa
    ))

    t_blue2 = np.hstack((
        np.vstack(( t[0, blue2], ix[0, blue2], ix[2, blue2])),  # noqa
        np.vstack((ix[2, blue2], ix[0, blue2],  t[1, blue2])),  # noqa
        np.vstack(( t[2, blue2], ix[2, blue2],  t[1, blue2])),  # noqa
    ))

    # new green elements
    t_green = np.hstack((
        np.vstack((t[1, green], ix[2, green], t[0, green])),
        np.vstack((t[2, green], ix[2, green], t[1, green])),
    ))

    # new nodes
    new_p = .5 * (p[:, facets[0, split_facets == 1]] +
                  p[:, facets[1, split_facets == 1]])

    return (
        np.hstack((p, new_p)),
        np.hstack((t[:, rest], t_red, t_blue1, t_blue2, t_green))
    )


def _tri_refine(p, t, marked):
    """Refine the set of marked elements."""

    t = _sort_tri(p, t)

    # build a list of facets and facet-to-triangle mappings
    facets = np.sort(np.hstack((
        t[[0, 1]],
        t[[1, 2]],
        t[[0, 2]],
    )), axis=0)
    tmp = np.ascontiguousarray(facets.T)
    tmp, ixa, ixb = np.unique(tmp.view([('', tmp.dtype)] * tmp.shape[1]),
                              return_index=True, return_inverse=True)

    facets = facets[:, ixa]
    t2f = ixb.reshape((3, t.shape[1]))
    split_facets = _find_facets(facets, t2f, marked)

    return _split_tris(p, t, facets, t2f, split_facets)


def refine(p, t, marked):

    if not isinstance(p, ndarray):
        p = np.array(p, dtype=np.float)

    if not isinstance(t, ndarray):
        t = np.array(t, dtype=np.int64)

    if p.shape[1] < p.shape[0]:
        p = p.T
        t = t.T

    if p.shape[0] == 2 and t.shape[0] == 3:
        return _tri_refine(p, t, marked)

    raise Exception("The provided mesh type not supported.")
