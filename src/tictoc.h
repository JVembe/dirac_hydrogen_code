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

static void toc_output(long *out)
{
    long s,u;
    gettimeofday(&te, NULL);
    s=te.tv_sec-tb.tv_sec;
    u=te.tv_usec-tb.tv_usec;
    *out = (s*1000000+u);
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
