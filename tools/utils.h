#ifndef _UTILS_H
#define _UTILS_H

#define HERE fprintf(stderr, "HERE %s:%d\n", __FUNCTION__, __LINE__);
#define MIN(a, b) ((a)<(b)?(a):(b))
#define MAX(a, b) ((a)>(b)?(a):(b))

#define ERROR(...)                                                    \
    {                                                                 \
        fprintf(stderr,  __VA_ARGS__);                                \
        exit(1);                                                      \
    }

#endif
