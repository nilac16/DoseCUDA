#ifndef MODEL_PARAMS_H
#define MODEL_PARAMS_H

#define VSADX 1340.0f // mm from iso
#define VSADY 1930.0f // mm from iso

#define AIR_DENSITY 0.0012f // g/cc
#define MU_CAL 4052.0f // calibration factor
#define PHOTON_MASS_ATTENUATION 0.042f //g / cm^2
#define PRIMARY_SOURCE_DISTANCE 1000.0f // mm from iso
#define SCATTER_SOURCE_DISTANCE 850.0f // mm from iso
#define MLC_DISTANCE 491.f // mm from iso
#define SCATTER_SOURCE_WEIGHT 0.12f
#define ELECTRON_MASS_ATTENUATION 0.8f // empirical factor - electron depth dose modeled as exponential
#define ELECTRON_WEIGHT 0.3f

#endif