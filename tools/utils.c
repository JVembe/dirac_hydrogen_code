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
        if(fabs(cimag(v1[i]-v2[i]))>1e-8) fprintf(stderr, "v1 %e v2 %e diff %e i\n",
                                                  cimag(v1[i]), cimag(v2[i]), cimag(v1[i]) - cimag(v2[i]));
        if(fabs(creal(v1[i]-v2[i]))>1e-8) fprintf(stderr, "v1 %e v2 %e diff %e\n",
                                                  creal(v1[i]), creal(v2[i]), creal(v1[i]) - creal(v2[i]));
    }
}
