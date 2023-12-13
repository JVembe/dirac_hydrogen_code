import numpy as npy
from scipy import sparse
from matplotlib import pyplot
import sys

idxtype = 'int32'

def csr_read(fname):
    with open(fname, 'r') as f:
        dim = npy.fromfile(f, dtype=idxtype, count=1)[0]
        nnz = npy.fromfile(f, dtype=idxtype, count=1)[0]
        indptr = npy.fromfile(f, dtype=idxtype, count=dim + 1)
        indices = npy.fromfile(f, dtype=idxtype, count=nnz)
        data = npy.fromfile(f, dtype='complex', count=nnz)
        global_dim = npy.max(indices)+1
        return sparse.csr_matrix((data, indices, indptr), shape=(dim, global_dim))

args = sys.argv[1:]
if len(args) == 0:
    args = ['H_part0.csr']
    # print('Usage: spplot [file name]')
    # exit(1)

fname = args[0]
res = csr_read(fname)
pyplot.spy(res, marker='.', markersize=1)
pyplot.show()
