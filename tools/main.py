import ctypes
import numpy as npy
from scipy import sparse
from matplotlib import pyplot
import metis

exec(open('output.py').read())

res = []
for i, idx in npy.ndenumerate(bdpam):
    if isinstance(idx, int):
        continue

    # only build the matrix non-zero structure here (1 instead of actuall value)
    Aidx = sparse.coo_matrix((npy.ones(idx[:,0].shape),(idx[:,1].astype('int'),idx[:,2].astype('int'))))
    bdpam_tmp = Aidx.tocsr();
    if isinstance(res, list):
        res = bdpam_tmp
    else:
        res = res + bdpam_tmp

print('non-zeros: ' + str(res.count_nonzero()))

# find non-zero columns, remove empty columns and rows
# structure is symmetric, so same rows and columns are empty
nzc=sum(res).nonzero()[1]
res=res[nzc,:]
res=res[:,nzc]

pyplot.spy(res, marker='.', markersize=1)
pyplot.show()

# metis partitioning - directly use the CSR storage from scipy sparse
idx_t = ctypes.c_int32
xadj = (idx_t*(res.indptr.shape[0]))()
xadj[:] = res.indptr[:]
adjncy = (idx_t*(res.indices.shape[0]))()
adjncy[:] = res.indices[:]
g = metis.METIS_Graph(idx_t(res.shape[0]), idx_t(1), xadj, adjncy, None, None, None)

# partition into nparts partitions (e.g., gpus / MPI ranks)
# original nodes are renumbered into the new numbers
#  - part contains the partition id of each node
#  - perm is a vector that maps new node numbers to old numbers (value = old number, index = new number)
nparts = 4
part = metis.part_graph(g, nparts)[1]
perm = npy.argsort(part, kind='stable')
res_metis = res[perm[:,None], perm]

pyplot.spy(res_metis, marker='.', markersize=1)
pyplot.show()


