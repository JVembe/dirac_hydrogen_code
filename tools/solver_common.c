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

#include "solver_common.h"
#include "types.h"
#include <math.h>
#include <stdlib.h>

#define max(a,b) ((a)>(b)?(a):(b))

// from spnrbasis.cpp
int ik(int i) {
    int ii = i/4;
    int abskappa = (int)(0.5*(sqrt(8.0*ii+1.0) - 1.0)) + 1;
    int sgnmod = max(4,abskappa*4);
    double sgnfloor = 2*abskappa * (abskappa - 1);
    int sgnkappa = ((i-sgnfloor)/sgnmod >= 0.5) - ((i-sgnfloor)/sgnmod < 0.5);
    return abskappa * sgnkappa;
}

double imu(int i) {
    int abskappa = abs(ik(i));
    int kmod = max(2,2*abskappa);
    double mu = i%kmod - abskappa + 0.5;
    return mu;
}

void beyondDipolePulse_init(beyondDipolePulse_t *this, double E0, double omega, double N)
{
    this->E0 = E0;
    this->omega = omega;
    this->T = (double)N*2*M_PI / omega;
}

void beoyndDipolePulse_axialPart(beyondDipolePulse_t *this, double t, cdouble_t *out)
{
    cdouble_t phi = CMPLX(-M_PI/2, 0);
    double E0 = this->E0;
    double omega = this->omega;
    double T = this->T;
    out[0] = ( -E0/(4*omega) * cos( phi + t * (2 * M_PI / T + omega)));
    out[1] = ( -E0/(4*omega) * sin( phi + t * (2 * M_PI / T + omega)));
    out[2] = ( -E0/(4*omega) * cos(-phi + t * (2 * M_PI / T - omega)));
    out[3] = (  E0/(4*omega) * sin(-phi + t * (2 * M_PI / T - omega)));
    out[4] = (  E0/(2*omega) * cos( phi + t * omega));
    out[5] = (  E0/(2*omega) * sin( phi + t * omega));
}
