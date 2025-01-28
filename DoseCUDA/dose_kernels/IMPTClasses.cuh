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

        /** @brief Compute the @b squared distance of voxel coordinates to their
         *      nearest point on a pencil beam
         * 	@param spot
         * 		The pencil beam
         * 	@param vox
         * 		The voxel coordinates in BEV
         */
        __device__ float caxDistance(const Spot &spot, const PointXYZ &vox);

};

class IMPTDose : public CudaDose{

    public:

        __host__ IMPTDose(DoseClass * h_dose);

};


//__global__ void pencilBeamKernel(IMPTDose * dose, IMPTBeam * beam);


void proton_raytrace_cuda(int gpu_id, DoseClass * h_dose, BeamClass  * h_beam);

void proton_spot_cuda(int gpu_id, DoseClass * h_dose, BeamClass  * h_beam);


#endif