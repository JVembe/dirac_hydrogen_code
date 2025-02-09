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

#include "superlu.h"
#include <slu_zdefs.h>
#include "utils.h"
#include "../src/tictoc.h"

void csr_coo2csr(const int *rowidx, const int *colidx, const doublecomplex *val, int dim, int nnz);

void write_ijk(const char *fname, int *Ai, int *Aj, doublecomplex *Ax, int nnz, int n)
{
    printf("write file %s\n", fname);
    FILE *fd = fopen(fname, "w+");
    if(!fd) ERROR("cant open %s\n", fname);

    // storage format: dim, nnz, Ai, Aj, Ax
    fwrite(&n, sizeof(int), 1, fd);
    fwrite(&nnz, sizeof(int), 1, fd);
    fwrite(Ai, sizeof(int), nnz, fd);
    fwrite(Aj, sizeof(int), nnz, fd);
    fwrite(Ax, sizeof(doublecomplex), nnz, fd);

    fclose(fd);
}

void slu_ijk_write(const char *fname, int *Ap, int *Aj, doublecomplex *Ax, int n)
{
    int nnz = Ap[n];
    int *Ai = (int *)malloc(sizeof(int)*nnz);
    for(int i=0; i<n; i++){
        for(int j=Ap[i]; j<Ap[i+1]; j++){
            Ai[j] = i;
        }
    }
    write_ijk(fname, Ai, Aj, Ax, nnz, n);
    free(Ai);
}


void slu_LU2coo(const SuperMatrix *L, const SuperMatrix *U,
                int_t **LAi_out, int_t **LAj_out, doublecomplex **LAx_out, int_t *Lnnz_out,
                int_t **UAi_out, int_t **UAj_out, doublecomplex **UAx_out, int_t *Unnz_out)
{
    SCformat     *Lstore;
    NCformat     *Ustore;
    int_t i, j, k, c, d, n, nsup, nnz;
    doublecomplex *dp;
    int *col_to_sup, *sup_to_col;
    int_t *rowind, *rowind_colptr;

    Lstore = (SCformat *) L->Store;
    int_t Lnnz = Lstore->nnz;
    int_t *LAi = (int *)malloc(sizeof(int)*Lnnz);
    int_t *LAj = (int *)malloc(sizeof(int)*Lnnz);
    int_t Lidx = 0;
    doublecomplex *LAx = (doublecomplex*)malloc(sizeof(doublecomplex)*Lnnz);

    Ustore = (NCformat *) U->Store;
    int_t Unnz = Ustore->nnz;
    int_t *UAi = (int *)malloc(sizeof(int)*Unnz);
    int_t *UAj = (int *)malloc(sizeof(int)*Unnz);
    int_t Uidx = 0;
    doublecomplex *UAx = (doublecomplex*)malloc(sizeof(doublecomplex)*Unnz);

    n = L->ncol;
    dp = (doublecomplex *) Lstore->nzval;
    col_to_sup = Lstore->col_to_sup;
    sup_to_col = Lstore->sup_to_col;
    rowind_colptr = Lstore->rowind_colptr;
    rowind = Lstore->rowind;
    for (k = 0; k <= Lstore->nsuper; ++k) {
        c = sup_to_col[k];
        nsup = sup_to_col[k+1] - c;
        for (j = c; j < c + nsup; ++j) {
            d = Lstore->nzval_colptr[j];
            for (i = rowind_colptr[c]; i < rowind_colptr[c+1]; ++i) {
                if(rowind[i]<=j){
                    UAi[Uidx] = rowind[i];
                    UAj[Uidx] = j;
                    UAx[Uidx] = dp[d];
                    Uidx++;

                    // Add diagonal 1 to L
                    if(rowind[i] == j){
                        LAi[Lidx] = rowind[i];
                        LAj[Lidx] = j;
                        LAx[Lidx] = (doublecomplex){1,0};
                        Lidx++;
                    }
                } else {
                    LAi[Lidx] = rowind[i];
                    LAj[Lidx] = j;
                    LAx[Lidx] = dp[d];
                    Lidx++;
                }
                d++;
            }
        }
    }

    for(int i=0; i<n; i++){
        for(int j=Ustore->colptr[i]; j<Ustore->colptr[i+1]; j++){
            UAj[Uidx] = i;
            UAi[Uidx] = Ustore->rowind[j];
            UAx[Uidx] = ((doublecomplex*)Ustore->nzval)[j];
            Uidx++;
        }
    }

    /* write_ijk("U.ijk", UAi, UAj, UAx, Uidx, n); */
    /* write_ijk("L.ijk", LAi, LAj, LAx, Lidx, n); */

    *LAi_out = LAi;
    *LAj_out = LAj;
    *LAx_out = LAx;
    *UAi_out = UAi;
    *UAj_out = UAj;
    *UAx_out = UAx;
    *Lnnz_out = Lidx;
    *Unnz_out = Uidx;
}

slu_matrix_t slu_create_matrix(int nrows, int ncols, int nnz,
                               void *Ax, int *Ai, int *Ap){
    SuperMatrix *A;
    NRformat *Astore;
    slu_matrix_t ret;
    A = malloc(sizeof(SuperMatrix));
    zCreate_CompCol_Matrix(A, nrows, ncols, nnz, (doublecomplex*)Ax, Ai, Ap, SLU_NC, SLU_Z, SLU_GE);
    Astore = A->Store;
    ret.ptr = A;
    return ret;
}

slu_LU_t slu_compute_ilu(slu_matrix_t opaqueA)
{
    SuperMatrix *A = (SuperMatrix*)opaqueA.ptr;
    NCformat *Ustore;
    SCformat *Lstore;

    int nrows = A->nrow, ncols = A->ncol;

    superlu_options_t options;
    int      *perm_c; /* column permutation vector */
    int      *perm_r; /* row permutations from partial pivoting */
    int      *etree;
    double   *R, *C;
    char     equed[1] = {'N'};
    SuperMatrix *L, *U;
    int_t    info, lwork;
    doublecomplex   *work = NULL;
    SuperMatrix B, X;
    double   rpg, rcond;
    GlobalLU_t     Glu; /* facilitate multiple factorizations with
                           SamePattern_SameRowPerm                  */
    mem_usage_t   *mem_usage = malloc(sizeof(mem_usage_t));
    SuperLUStat_t *stat = malloc(sizeof(SuperLUStat_t));;
    slu_LU_t ret = {0};

    L = malloc(sizeof(SuperMatrix));
    U = malloc(sizeof(SuperMatrix));

    if ( !( etree = int32Malloc(nrows)) ) ABORT("Malloc fails for etree[].");
    if ( !(perm_r = int32Malloc(ncols)) ) ABORT("Malloc fails for perm_r[].");
    if ( !(perm_c = int32Malloc(nrows)) ) ABORT("Malloc fails for perm_c[].");
    if ( !(R = (double *) SUPERLU_MALLOC(A->nrow * sizeof(double))) )
        ABORT("SUPERLU_MALLOC fails for R[].");
    if ( !(C = (double *) SUPERLU_MALLOC(A->ncol * sizeof(double))) )
        ABORT("SUPERLU_MALLOC fails for C[].");
    info = 0;

    // this is abuse: we need the vectors only because SuperLU demands them,
    // but is not using them - we do not solve.
    zCreate_Dense_Matrix(&B, nrows, 1, NULL, nrows, SLU_DN, SLU_Z, SLU_GE);
    zCreate_Dense_Matrix(&X, nrows, 1, NULL, nrows, SLU_DN, SLU_Z, SLU_GE);

    /* Initialize the statistics variables. */
    StatInit(stat);

    /* Set the default input options:
       options.Fact = DOFACT;
       options.Equil = YES;
       options.ColPerm = COLAMD;
       options.DiagPivotThresh = 0.1; //different from complete LU
       options.Trans = NOTRANS;
       options.IterRefine = NOREFINE;
       options.SymmetricMode = NO;
       options.PivotGrowth = NO;
       options.ConditionNumber = NO;
       options.PrintStat = YES;
       options.RowPerm = LargeDiag_MC64;
       options.ILU_DropTol = 1e-4;
       options.ILU_FillTol = 1e-2;
       options.ILU_FillFactor = 10.0;
       options.ILU_DropRule = DROP_BASIC | DROP_AREA;
       options.ILU_Norm = INF_NORM;
       options.ILU_MILU = SILU;
    */
    ilu_set_default_options(&options);

    /* Modify the defaults. */
    options.Equil = NO;
    options.ColPerm = NATURAL;
    options.RowPerm = NO;
    options.PivotGrowth = YES;    /* Compute reciprocal pivot growth */
    options.ConditionNumber = YES;/* Compute reciprocal condition number */
    options.ILU_DropRule = DROP_BASIC;
    options.ILU_DropTol = 1e-6;
    options.ILU_MILU = SILU; //SMILU_1;
    options.DiagPivotThresh = 0;
    options.ILU_FillFactor = 2.0;

    B.ncol = 0;  /* not to perform triangular solution */

    tic(); PRINTF0("compute ilu "); fflush(stdout);
#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    zgsisx(&options, A, perm_c, perm_r, etree, equed, R, C, L, U, NULL, 0,
           &B, &X, &rpg, &rcond, &Glu, mem_usage, stat, &info);    
#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    toc();
#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    Ustore = U->Store;
    Lstore = L->Store;
    printf("SuperLU L dimension %dx%d; # nonzeros %d\n", (int)L->nrow, (int)L->ncol, (int)Lstore->nnz);
    printf("SuperLU U dimension %dx%d; # nonzeros %d\n", (int)U->nrow, (int)U->ncol, (int)Ustore->nnz);
    fflush(stdout);

    ret.A = A;
    ret.U = U;
    ret.L = L;
    ret.perm_c = perm_c;
    ret.perm_r = perm_r;
    ret.R = R;
    ret.C = C;
    ret.etree = etree;
    ret.mem_usage = mem_usage;
    ret.stat = stat;
    ret.equed = equed[0];

    return ret;
}

void slu_lu_solve(slu_LU_t *lu, doublecomplex *rhs, doublecomplex *x)
{
    int nrows = ((SuperMatrix*)lu->A)->nrow;
    superlu_options_t options;
    int_t    info = 0;

    static DNformat X, Y;
    static SuperMatrix XX = {SLU_DN, SLU_Z, SLU_GE, 1, 1, &X};
    static SuperMatrix YY = {SLU_DN, SLU_Z, SLU_GE, 1, 1, &Y};
    XX.nrow = YY.nrow = nrows;
    X.lda = Y.lda = nrows;
    X.nzval = x;
    Y.nzval = rhs;

    ilu_set_default_options(&options);

    /* Set the options to do solve-only. */
    options.Fact = FACTORED;
    zgsisx(&options, lu->A, lu->perm_c, lu->perm_r, NULL, &lu->equed, lu->R, lu->C, lu->L, lu->U, NULL, 0,
           &YY, &XX, NULL, NULL, NULL, lu->mem_usage, lu->stat, &info);

    // cuda: cusparseSpSV
}
