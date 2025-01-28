/*
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
*/

#ifndef _GPU_SOLVER_H
#define _GPU_SOLVER_H

#ifdef __cplusplus
extern "C" {
#endif
       
    void gpu_init_model_matrices(int nmatrices, const sparse_csr_t *cpu_g,
                                  const sparse_csr_t *cpu_gt, const sparse_csr_t *cpu_h0);
    void gpu_compute_row_col(int lmax, const sparse_csr_t *H,
                              sparse_csr_t *Hfull_blk, sparse_csr_t *Hfull);
    void gpu_compute_timedep_matrices(double h, double dt, csr_data_t *ft, int lmax,
                                       const sparse_csr_t *Hfull_blk, sparse_csr_t *Hfull,
                                       gpu_sparse_csr_t *gpu_Hfull);
    
#ifdef __cplusplus
}
#endif

#endif /* _GPU_SOLVER_H */
