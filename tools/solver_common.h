#ifndef _SOLVER_COMMON_H
#define _SOLVER_COMMON_H

// simple C-fied beyondDipolePulse implementation
typedef struct
{
    double E0;
    double omega;
    double T;
} beyondDipolePulse_t;

#define SoL 137.035999084

int ik(int i);
double imu(int i);

#endif /* _SOLVER_COMMON_H */
