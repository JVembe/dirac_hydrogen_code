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
