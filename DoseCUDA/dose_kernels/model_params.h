#ifndef MODEL_PARAMS_H
#define MODEL_PARAMS_H

#define VSADX 1340.0f // mm from iso
#define VSADY 1930.0f // mm from iso

#define AIR_DENSITY 0.0012f // g/cc
#define MU_CAL 1.776e-2f
#define PHOTON_MASS_ATTENUATION 0.042f //g / cm^2
#define PRIMARY_SOURCE_DISTANCE 1000.0f // mm from iso
#define SCATTER_SOURCE_DISTANCE 850.0f // mm from iso
#define MLC_DISTANCE 643.2f // mm from iso
#define SCATTER_SOURCE_WEIGHT 0.33f
#define ELECTRON_MASS_ATTENUATION 2.0f // empirical factor - electron depth dose modeled as exponential
#define ELECTRON_WEIGHT 0.65f

#endif