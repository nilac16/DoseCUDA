#ifndef IMPTCLASSES_H
#define IMPTCLASSES_H

#include "./CudaClasses.cuh"


#define TILE_WIDTH 4

class IMPTBeam : public CudaBeam{

    public:

        __host__ IMPTBeam(BeamClass * h_beam);

        __device__ void interpolateProtonLUT(float wet, float * idd, float * sigma, size_t layer_id);

        __device__ float sigmaAir(float wet, float distance_to_source, size_t layer_id);

        __device__ void nuclearHalo(const float wet, float * halo_sigma, float * halo_weight, size_t layer_id);

};

class IMPTDose : public CudaDose{

    public:

        __host__ IMPTDose(DoseClass * h_dose);

};


__global__ void rayTraceKernel(IMPTDose * dose, IMPTBeam * beam);

__global__ void pencilBeamKernel(IMPTDose * dose, IMPTBeam * beam);


void proton_raytrace_cuda(int gpu_id, DoseClass * h_dose, BeamClass  * h_beam);

void proton_spot_cuda(int gpu_id, DoseClass * h_dose, BeamClass  * h_beam);


#endif