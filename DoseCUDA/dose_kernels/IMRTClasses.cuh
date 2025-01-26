#ifndef IMRTCLASSES_H
#define IMRTCLASSES_H

#include "./CudaClasses.cuh"


#define TILE_WIDTH 4

const float h_kernel[6][6] = {
		{1.875,		20.625,		43.125,		61.875,		88.125,		106.875},
		{1.280,		0.603,		0.183, 		0.848e-1, 	0.158e-1, 	0.145e-1},
		{1.910, 	1.830, 		2.050, 		2.850, 		3.710, 		7.710},
		{0.124e-1, 	0.666e-2, 	0.212e-2, 	0.954e-3, 	0.450e-3, 	0.333e-3},
		{2.810e-1, 	0.253e-1, 	0.291e-1, 	0.244e-1, 	0.184e-1, 	0.185e-1},
		{3.000, 	2.000, 		2.000, 		2.000, 		2.000, 		1.000}}; //this can be constant for all 6 MV photon machines, but needs to be updated for other photon energies

const float h_attenuation_coefficients[12] = {0.09687, 0.07072, 0.05754, 0.04942, 0.04385, 0.03969, 0.03637, 0.03403, 0.03230, 0.03031, 0.02905, 0.02770};
const float h_energy_fluence[12] = {0.18004762, 0.18004762, 0.0682789,  0.03990357, 0.0403678,  0.02817274, 0.0261157,  0.02836988, 0.03217569, 0.10721887, 0.12699279, 0.14230882};
const float h_scatter_energy_fluence[12] = {-0.49326286,  0.16015799,  0.29606239,  0.23461438,  0.299726,    0.25019492, 0.20748785,  0.09246743,  0.0702758,  -0.01659201, -0.05079632, -0.05033556};

const float h_profile[] = 						{1.000, 1.005, 1.009, 1.014, 1.016, 1.011, 1.004, 0.993, 0.985, 0.985,
												0.985, 0.985, 0.985, 0.875, 0.780, 0.692, 0.050, 0.010, 0.001, 0.001};

const float h_soften[] = 						{1.000, 1.040, 1.080, 1.120, 1.160, 1.200, 1.240, 1.300, 1.320, 1.360,
												1.420, 1.440, 1.460, 1.480, 1.490, 1.500, 1.520, 1.560, 1.590, 1.600};

const float h_radius[] = 						{0.000, 20.00, 40.00, 60.00, 80.00, 100.0, 120.0, 150.0, 160.0, 180.0,
												210.0, 220.0, 230.0, 240.0, 245.0, 250.0, 260.0, 280.0, 295.0, 300.0};


__constant__ float g_kernel[6][6]; //TERMA to dose kernel
__constant__ float g_attenuation_coefficients[12]; //attenuation coefficients
__constant__ float g_energy_fluence[12]; //energy fluence
__constant__ float g_scatter_energy_fluence[12]; //scatter energy fluence
__constant__ float g_profile[20]; //off axis intensity
__constant__ float g_soften[20]; //off axis softening factor
__constant__ float g_radius[20]; //radius of profile/softening points


__host__ __device__ float interp(float x, const float * xd, const float * yd, const int n);

class IMRTBeam : public CudaBeam{

    public:

        __host__ IMRTBeam(BeamClass * h_beam);

        __device__ float headTransmission(const PointXYZ * point_xyz, const float distance_to_source, const float xSigma, const float ySigma);

		__device__ void offAxisFactors(const PointXYZ * point_xyz, float * off_axis_factor, float * off_axis_softening);

};

class IMRTDose : public CudaDose{

    public:

        __host__ IMRTDose(DoseClass * h_dose);

};


// __global__ void termaKernel(IMRTDose * dose, IMRTBeam * beam);

// __global__ void cccKernel(IMRTDose * dose, IMRTBeam * beam);


void photon_dose_cuda(int gpu_id, DoseClass * h_dose, BeamClass * h_beam);


#endif