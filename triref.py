import numpy as np

def rgb(p, t, marked):
    """Perform adaptive RGB refinement.
    This is more or less directly ported from the book of Sören Bartels."""
    # change (0,2) to longest edge
    for itr in range(t.shape[1]):
        l01 = np.sqrt(np.sum((p[:, t[0, itr]] - p[:, t[1, itr]])**2))
        l12 = np.sqrt(np.sum((p[:, t[1, itr]] - p[:, t[2, itr]])**2))
        l02 = np.sqrt(np.sum((p[:, t[0, itr]] - p[:, t[2, itr]])**2))
        if l02 > l01 and l02 > l12:
            # OK. (0 2) is longest
            continue
        elif l01 > l02 and l01 > l12:
            # (0 1) is longest
            tmp = t[2, itr]
            t[2, itr] = t[1, itr]
            t[1, itr] = tmp
        elif l12 > l01 and l12 > l02:
            # (1 2) is longest
            tmp = t[0, itr]
            t[0, itr] = t[1, itr]
            t[1, itr] = tmp

    n4e = t.T
    c4n = p.T

    # edges and edge mappings
    edges = np.reshape(n4e[:,[1,2,0,2,0,1]].T, (2,-1)).T

    edges = np.sort(edges,axis=1)

    tmp = np.ascontiguousarray(edges)
    xoxo, ixa, ixb = np.unique(tmp.view([('', tmp.dtype)] * tmp.shape[1]),
                              return_index=True, return_inverse=True)
    edges = edges[ixa]
    el2edges = ixb.reshape((3, -1)).T

    nEdges = edges.shape[0]
    nC = c4n.shape[0]
    tmp = 1

    markedEdges = np.zeros(nEdges)
    #print el2edges.shape
    markedEdges[np.reshape(el2edges[marked], (-1, 1))] = 1

    while tmp > 0:
        tmp = np.count_nonzero(markedEdges)
        el2markedEdges = markedEdges[el2edges]
        el2markedEdges[el2markedEdges[:,0] + el2markedEdges[:,2]>0,1] = 1.0
        markedEdges[el2edges[el2markedEdges==1.0]] = 1
        tmp = np.count_nonzero(markedEdges)-tmp

    newNodes = np.zeros(nEdges)-1
    newNodes[markedEdges==1.0] = np.arange(np.count_nonzero(markedEdges)) + nC
    newInd = newNodes[el2edges]

    red = (newInd[:,0]>=0) & (newInd[:,1] >= 0) & (newInd[:,2] >= 0)
    blue1 = (newInd[:,0]>=0) & (newInd[:,1] >= 0) & (newInd[:,2] == -1)
    blue3 = (newInd[:,0]==-1) & (newInd[:,1] >= 0) & (newInd[:,2] >= 0)
    green = (newInd[:,0]==-1) & (newInd[:,1] >= 0) & (newInd[:,2] == -1)
    remain = (newInd[:,0]==-1) & (newInd[:,1] == -1) & (newInd[:,2] == -1)

    n4e_red = np.vstack((n4e[red,0],newInd[red,2],newInd[red,1]))
    n4e_red_1 = np.reshape(n4e_red,(3,-1)).T
    n4e_red = np.vstack((newInd[red,1],newInd[red,0],n4e[red,2]))
    n4e_red_2 = np.reshape(n4e_red,(3,-1)).T
    n4e_red = np.vstack((newInd[red,2],n4e[red,1],newInd[red,0]))
    n4e_red_3 = np.reshape(n4e_red,(3,-1)).T
    n4e_red = np.vstack((newInd[red,0],newInd[red,1],newInd[red,2]))
    n4e_red_4 = np.reshape(n4e_red,(3,-1)).T
    n4e_red = np.vstack((n4e_red_1, n4e_red_2, n4e_red_3, n4e_red_4))

    n4e_blue1 = np.vstack((n4e[blue1,1],newInd[blue1,1],n4e[blue1,0]))
    n4e_blue1_1 = np.reshape(n4e_blue1,(3,-1)).T
    n4e_blue1 = np.vstack((n4e[blue1,1],newInd[blue1,0],newInd[blue1,1]))
    n4e_blue1_2 = np.reshape(n4e_blue1,(3,-1)).T
    n4e_blue1 = np.vstack((newInd[blue1,1],newInd[blue1,0],n4e[blue1,2]))
    n4e_blue1_3 = np.reshape(n4e_blue1,(3,-1)).T
    n4e_blue1 = np.vstack((n4e_blue1_1, n4e_blue1_2, n4e_blue1_3))

    n4e_blue3 = np.vstack((n4e[blue3,0],newInd[blue3,2],newInd[blue3,1]))
    n4e_blue3_1 = np.reshape(n4e_blue3,(3,-1)).T
    n4e_blue3 = np.vstack((newInd[blue3,1],newInd[blue3,2],n4e[blue3,1]))
    n4e_blue3_2 = np.reshape(n4e_blue3,(3,-1)).T
    n4e_blue3 = np.vstack((n4e[blue3,2],newInd[blue3,1],n4e[blue3,1]))
    n4e_blue3_3 = np.reshape(n4e_blue3,(3,-1)).T
    n4e_blue3 = np.vstack((n4e_blue3_1, n4e_blue3_2, n4e_blue3_3))

    n4e_green = np.vstack((n4e[green,1],newInd[green,1],n4e[green,0],
        n4e[green,2],newInd[green,1],n4e[green,1]))
    n4e_green = np.reshape(n4e_green,(3,-1)).T

    n4e = np.vstack((n4e[remain], n4e_red, n4e_blue1, n4e_blue3, n4e_green))

    newCoord = .5*(c4n[edges[markedEdges==1.0,0]]
            + c4n[edges[markedEdges==1.0,1]])

    c4n = np.vstack((c4n, newCoord))

    return c4n.T, n4e.T.astype(np.int64)
