'''
	GaDE - Gpu-accelerated solver for the time dependent Dirac Equation
	
    Copyright (C) 2025  Johanne Elise Vembe <johannevembe@gmail.com>
    Copyright (C) 2025  Marcin Krotkiewski <marcink@uio.no>
	Copyright (C) 2025  Hicham Agueny <hicham.agueny@uib.no>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published
    by the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''

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
