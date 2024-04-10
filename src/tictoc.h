#ifndef TICTOC_H
#define TICTOC_H

#include <sys/time.h>
#include <time.h>

static struct timeval tb, te;
extern int rank;

static void tic()
{
    gettimeofday(&tb, NULL);
}

#ifndef PRINTF0
#define PRINTF0 printf
#endif
static void toc()
{
    long s,u;
    gettimeofday(&te, NULL);
    s=te.tv_sec-tb.tv_sec;
    u=te.tv_usec-tb.tv_usec;
    PRINTF0("time:\t\t%li.%.6lis\n", (s*1000000+u)/1000000, (s*1000000+u)%1000000);
    fflush(stdout);
}

#endif
