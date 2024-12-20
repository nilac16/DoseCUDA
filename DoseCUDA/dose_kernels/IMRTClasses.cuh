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
		{10.000, 	5.000, 		5.000, 		5.000, 		5.000, 		2.000}}; //this can be constant for all 6 MV photon machines, but needs to be updated for other photon energies

__constant__ float g_kernel[6][6]; //TERMA to dose kernel


class IMRTBeam : public CudaBeam{

    public:

        __host__ IMRTBeam(BeamClass * h_beam);

        __device__ float headTransmission(const PointXYZ * point_xyz);

};

class IMRTDose : public CudaDose{

    public:

        __host__ IMRTDose(DoseClass * h_dose);

};

__global__ void rayTraceKernel(IMRTDose * dose, IMRTBeam * beam);

__global__ void termaKernel(IMRTDose * dose, IMRTBeam * beam);

// __global__ void cccKernel(IMRTDose * dose, IMRTBeam * beam);


void photon_dose_cuda(int gpu_id, DoseClass * h_dose, BeamClass * h_beam);


#endif