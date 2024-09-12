#ifndef TICTOC_H
#define TICTOC_H

#include <sys/time.h>
#include <time.h>

static struct timeval tb, te, tc,tf;
extern int rank;

static void tic()
{
    gettimeofday(&tb, NULL);
}

static void bigtic()
{
    gettimeofday(&tc, NULL);
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

static void bigtoc()
{
    long s,u;
    gettimeofday(&tf, NULL);
    s=tf.tv_sec-tc.tv_sec;
    u=tf.tv_usec-tc.tv_usec;
    PRINTF0("time:\t\t%li.%.6lis\n", (s*1000000+u)/1000000, (s*1000000+u)%1000000);
    fflush(stdout);
}

#endif
