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

#include "types.h"
#include "utils.h"
#include <math.h>
#include <complex.h>

void compare_vectors(csr_data_t *v1, csr_data_t *v2, csr_index_t dim)
{
    // validate - compare two host vectors
    for(int i=0; i<dim; i++) {
        if(isnan(cimag(v1[i]+v2[i])) || isnan(creal(v1[i]+v2[i]))) {
            printf("nan in vector!\n");
            continue;
        }
        double norm;
        norm = fabs(fmax(cimag(v1[i]), cimag(v2[i])));
        if(fabs(cimag(v1[i]-v2[i]))/norm>1e-11)
            fprintf(stderr, "v1 %e v2 %e diff %e i\n",
                    cimag(v1[i]), cimag(v2[i]), cimag(v1[i]) - cimag(v2[i]));
        norm = fabs(fmax(creal(v1[i]), creal(v2[i])));
        if(fabs(creal(v1[i]-v2[i]))/norm>1e-11)
            fprintf(stderr, "v1 %e v2 %e diff %e\n",
                    creal(v1[i]), creal(v2[i]), creal(v1[i]) - creal(v2[i]));
    }
}
