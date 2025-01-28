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

#ifndef _SOLVER_COMMON_H
#define _SOLVER_COMMON_H

// simple C-fied beyondDipolePulse implementation
typedef struct
{
    double E0;
    double omega;
    double T;
} beyondDipolePulse_t;

#ifndef M_PI
#define M_PI (3.14159265358979323846)
#endif

#define SoL 137.035999084

int ik(int i);
double imu(int i);

#endif /* _SOLVER_COMMON_H */
