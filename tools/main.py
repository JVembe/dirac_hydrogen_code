import glob
import ctypes
import numpy as npy
from scipy import sparse
from matplotlib import pyplot
import metis

idxtype = 'int32'

def csr_read(fname):
    with open(fname, 'r') as f:
        dim = npy.fromfile(f, dtype=idxtype, count=1)[0]
        print([fname, ' dim ', dim]);
        nnz = npy.fromfile(f, dtype=idxtype, count=1)[0]
        indptr = npy.fromfile(f, dtype=idxtype, count=dim + 1)
        indices = npy.fromfile(f, dtype=idxtype, count=nnz)
        data = npy.fromfile(f, dtype='complex', count=nnz)
        return sparse.csr_matrix((data, indices, indptr), shape=(dim, dim))


# the following loop produces the non-zero structure of the global couplings matrix
# individual H matrices still have to be saved in separate files to compute the submatrices
fnames = glob.glob('H0l*csr')
lmax = len(fnames)
if lmax == 0:
    print('Did not find any H*csr files, bailing out')
    exit(1)
print('lmax: ' + str(lmax))
res = []
for l in range(0, lmax):
    fname = 'H0l' + str(l) + '.csr'
    if isinstance(res, list):
        res = csr_read(fname)
    else:
        res = res + csr_read(fname)
    fname = 'H1l' + str(l) + '.csr'
    res = res + csr_read(fname)


res_orig = res
# remove empty rows and columns from the full coupling nnz pattern
# find non-zero columns, remove empty columns and rows
# structure is symmetric, so same rows and columns are empty
nzc = sum(res).nonzero()[1]
res = res[nzc[:, None], nzc]

# add diagonal entries
res = res + sparse.identity(res.shape[0],format="csr")
res_orig = res_orig + sparse.identity(res_orig.shape[0],format="csr")

part_Ap = npy.array([0, len(nzc)], dtype=idxtype)
nparts = 4
if nparts > 1:
    # metis partitioning - for now only for demonstration
    # directly use the CSR storage from scipy sparse
    idx_t = ctypes.c_int32
    xadj = (idx_t * (res.indptr.shape[0]))()
    xadj[:] = res.indptr[:]
    adjncy = (idx_t * (res.indices.shape[0]))()
    adjncy[:] = res.indices[:]
    g = metis.METIS_Graph(idx_t(res.shape[0]), idx_t(1), xadj, adjncy, None, None, None)

    # partition into nparts partitions (e.g., gpus / MPI ranks)
    # original nodes are renumbered into the new numbers
    #  - part contains the partition id of each node
    #  - perm is a vector that maps new node numbers to old numbers (perm[new id] = old id)
    part = metis.part_graph(g, nparts)[1]
    part = npy.array(part, idxtype)
    perm = npy.argsort(part, kind='stable')
    perm = perm.astype(idxtype)
    nzc = nzc[perm]
    res = res[perm[:, None], perm]

    # find row-based partitioning
    part = part[perm]
    part_id, part_count = npy.unique(part, return_counts=True)
    part_count = part_count.astype(idxtype)
    part_Ap = npy.cumsum(npy.concatenate((0, part_count), axis=None), dtype=idxtype)

    pyplot.spy(res, marker='.', markersize=1)
    pyplot.show()

res = res_orig[nzc[:, None], nzc]

pyplot.spy(res, marker='.', markersize=1)
pyplot.show()

with open('H.csr', 'w+') as f:
    nnz = npy.array(res.count_nonzero(), idxtype)
    dim = npy.array(res.shape[1], idxtype)
    dim.tofile(f)
    nnz.tofile(f)
    res.indptr.tofile(f)
    res.indices.tofile(f)
    res.data.tofile(f)
    # write the map of new to original row/col indices
    nzc.tofile(f)

    # write the number of parallel partitions, and the partitioning
    npy.array(len(part_Ap)-1, dtype=idxtype).tofile(f)
    part_Ap.tofile(f)
