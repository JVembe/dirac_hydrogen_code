#ifndef _UTILS_H
#define _UTILS_H

#include <stdio.h>
#include <stdlib.h>

#define PRINTF0(...) if(rank==0) printf(__VA_ARGS__)

#define HERE fprintf(stderr, "HERE %s:%d\n", __FUNCTION__, __LINE__);
#define MIN(a, b) ((a)<(b)?(a):(b))
#define MAX(a, b) ((a)>(b)?(a):(b))

#define ERROR(...)                              \
    {                                           \
        fprintf(stderr,  __VA_ARGS__);          \
        exit(1);                                \
    }

#ifdef USE_MPI
#define CHECK_MPI(func) {                                       \
        int status = (func);                                    \
        if (status != 0) {                                      \
            char string[256];                                   \
            int resultlen;                                      \
            MPI_Error_string(status, string, &resultlen);       \
            printf("MPI failed at line %s:%d error: %s\n",      \
                   __FUNCTION__, __LINE__, string);             \
            exit(1);                                            \
        }                                                       \
    }
#endif

#endif
