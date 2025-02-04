#ifndef IMRTCLASSES_H
#define IMRTCLASSES_H

#include "CudaClasses.cuh"

typedef struct {

    float x1;
    float x2;
    float y_offset;
    float y_width;

} MLCPair;

__host__ __device__ float interp(float x, const float * xd, const float * yd, const int n);

class IMRTBeam : public CudaBeam{

    public:

        struct Model {

            float air_density;  // g/cc
            float mu_cal;       // calibration factor

            float primary_src_dist; // mm from iso
            float scatter_src_dist; // mm from iso

            float mlc_distance;     // mm from iso

            float scatter_src_weight;
            float electron_mass_attenuation;    // empirical factor - electron depth dose modeled as exponential

        } model;

		float collimator_angle;
        float mu;

		float sinca, cosca; // Cached collimator angle trig functions

		MLCPair * mlc;  // MLC leaf pairs
        int n_mlc_pairs;    // Number of leaf pairs

		float * off_axis_radii;
        float * off_axis_factors;
        float * off_axis_softening;
        int off_axis_len;

        __host__ IMRTBeam(IMRTBeam * h_beam);
		__host__ IMRTBeam(float * iso, float gantry_angle, float couch_angle, float collimator_angle, const Model * model);

        __device__ float headTransmission(const PointXYZ * point_xyz, const float distance_to_source, const float xSigma, const float ySigma);

		__device__ void offAxisFactors(const PointXYZ * point_xyz, float * off_axis_factor, float * off_axis_softening);

		/** Takes collimator angle into account */
		__device__ void pointXYZImageToHead(const PointXYZ * point_img, PointXYZ * point_head);

		/** Takes collimator angle into account */
        __device__ void pointXYZHeadToImage(const PointXYZ * point_head, PointXYZ * point_img);

};

class IMRTDose : public CudaDose{

    public:

		using CudaDose::CudaDose;

        __host__ IMRTDose(CudaDose * h_dose);

};

__global__ void termaKernel(IMRTDose * dose, IMRTBeam * beam, float * TERMAArray, float * ElectronArray);
__global__ void cccKernel(IMRTDose * dose, IMRTBeam * beam, Texture3D TERMATexture, Texture3D DensityTexture, float * ElectronArray);

void photon_dose_cuda(int gpu_id, IMRTDose * h_dose, IMRTBeam * h_beam);

#endif