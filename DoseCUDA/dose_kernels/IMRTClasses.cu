#include "./IMRTClasses.cuh"
#ifndef DOSECUDA_DEVICE_POINTER
#	define DOSECUDA_DEVICE_POINTER
#endif
#include "MemoryClasses.h"


__host__ IMRTBeam::IMRTBeam(BeamClass * h_beam) : CudaBeam(h_beam) {}

__device__ float IMRTBeam::headTransmission(const PointXYZ * point_xyz){
	
	PointXYZ point_xyz_div;
	point_xyz_div.x = -point_xyz->x / ((-point_xyz->y + PRIMARY_SOURCE_DISTANCE) / PRIMARY_SOURCE_DISTANCE);
	point_xyz_div.z = point_xyz->z / ((-point_xyz->y + PRIMARY_SOURCE_DISTANCE) / PRIMARY_SOURCE_DISTANCE);
	point_xyz_div.y = point_xyz->y;
	
	float transmission = 0.0;
	float tip_distance = 0.0;
	float x1_eff, x2_eff;

	for(int i=0; i < this->n_mlc_pairs; i++){
		if((point_xyz_div.z > (this->mlc[i].y_offset - (this->mlc[i].y_width / 2.0))) && (point_xyz_div.z <= (this->mlc[i].y_offset + (this->mlc[i].y_width / 2.0)))){
			x1_eff = this->mlc[i].x1 - 2.0;
			x2_eff = this->mlc[i].x2 + 2.0;
			if(point_xyz_div.x > x1_eff && point_xyz_div.x < x2_eff){
				tip_distance = fminf(fabsf(point_xyz_div.x - x1_eff), fabsf(point_xyz_div.x - x2_eff));
				transmission = (1.0 - expf(-tip_distance));
			}
			break;
		}
	}

	return transmission;

}


__host__ IMRTDose::IMRTDose(DoseClass * h_dose) : CudaDose(h_dose) {}


__global__ void rayTraceKernelPhoton(IMRTDose * dose, IMRTBeam * beam){

	PointIJK vox_ijk;
	vox_ijk.k = threadIdx.x + (blockIdx.x * blockDim.x);
	vox_ijk.j = threadIdx.y + (blockIdx.y * blockDim.y);
	vox_ijk.i = threadIdx.z + (blockIdx.z * blockDim.z);

	if(!dose->pointIJKWithinImage(&vox_ijk)) {
		return;
	}

	size_t vox_index = dose->pointIJKtoIndex(&vox_ijk);

	PointXYZ vox_xyz;
	dose->pointIJKtoXYZ(&vox_ijk, &vox_xyz, beam);

	PointXYZ uvec;
	beam->unitVectorToSource(&vox_xyz, &uvec);

	PointXYZ vox_ray_xyz;
	PointIJK vox_ray_ijk;

	int vox_ray_index = 0;
    float ray_length = 0.0;
    float wet_sum = -0.05;
    float density = 0.0;
	const float step_length = 1.0;

    for(int i=0; i<360; i++){

		vox_ray_xyz.x = fmaf(uvec.x, ray_length, vox_xyz.x);
		vox_ray_xyz.y = fmaf(uvec.y, ray_length, vox_xyz.y);
		vox_ray_xyz.z = fmaf(uvec.z, ray_length, vox_xyz.z);

		dose->pointXYZtoIJK(&vox_ray_xyz, &vox_ray_ijk, beam);

		if(!dose->pointIJKWithinImage(&vox_ray_ijk)){
			break;
		}

		vox_ray_index = dose->pointIJKtoIndex(&vox_ray_ijk);

		density = dose->DensityArray[vox_ray_index];

		wet_sum = fmaf(fmaxf(density, 0.0), step_length / 10.0, wet_sum);

		ray_length += step_length;

	}

	dose->WETArray[vox_index] = wet_sum;

    __syncthreads();

}

__global__ void termaKernel(IMRTDose * dose, IMRTBeam * beam, float * TERMAArray){

	PointIJK vox_ijk;
	vox_ijk.k = threadIdx.x + (blockIdx.x * blockDim.x);
	vox_ijk.j = threadIdx.y + (blockIdx.y * blockDim.y);
	vox_ijk.i = threadIdx.z + (blockIdx.z * blockDim.z);

	if(!dose->pointIJKWithinImage(&vox_ijk)) {
		return;
	}

	size_t vox_index = dose->pointIJKtoIndex(&vox_ijk);

	PointXYZ vox_xyz, vox_head_xyz;
	dose->pointIJKtoXYZ(&vox_ijk, &vox_xyz, beam);
	dose->pointXYZImageToHead(&vox_xyz, &vox_head_xyz, beam);
	
	float distance_to_source = beam->distanceToSource(&vox_xyz);
	float transmission = beam->headTransmission(&vox_head_xyz);
	float wet = dose->WETArray[vox_index];

	TERMAArray[vox_index] = transmission * powf(PRIMARY_SOURCE_DISTANCE / distance_to_source, 2.0) * (expf(-PHOTON_MASS_ATTENUATION * wet));

    __syncthreads();

}

__global__ void cccKernel(IMRTDose * dose, IMRTBeam * beam, float * TERMAArray){

	PointIJK vox_ijk;
	vox_ijk.k = threadIdx.x + (blockIdx.x * blockDim.x);
	vox_ijk.j = threadIdx.y + (blockIdx.y * blockDim.y);
	vox_ijk.i = threadIdx.z + (blockIdx.z * blockDim.z);

	if(!dose->pointIJKWithinImage(&vox_ijk)) {
		return;
	}

	size_t vox_index = dose->pointIJKtoIndex(&vox_ijk);

	PointXYZ vox_xyz;
	dose->pointIJKtoXYZ(&vox_ijk, &vox_xyz, beam);

	PointXYZ uvec;
	beam->unitVectorToSource(&vox_xyz, &uvec);

	float dose_value = 0.0;
	float xc, yc, zc, xr, yr, zr, Rs, Rp, ray_length, sp, th, Am, am, Bm, bm, Ti, Di;
	PointXYZ vox_ray_xyz;
	PointIJK vox_ray_ijk;
	int vox_ray_index;

	for(int i = 0; i < 6; i+=2){
		for(int j = 0; j < 12; j+=2){

			//Kernel constants
			th = g_kernel[0][i];
			Am = g_kernel[1][i];
			am = g_kernel[2][i];
			Bm = g_kernel[3][i];
			bm = g_kernel[4][i];
			ray_length = g_kernel[5][i];

			//kernel vector
			xc = sinf(th * M_PI / 180.0) * cosf((float)j * 30.0 * M_PI / 180.0);
			yc = sinf(th * M_PI / 180.0) * sinf((float)j * 30.0 * M_PI / 180.0);
			zc = cosf(th * M_PI / 180.0);

			//kernel tilting
			if(uvec.y > 0){
				xr =  (xc * (1 - (powf(uvec.x, 2.0) / (1 + uvec.y)))) 	- (yc * ((uvec.x * uvec.z) / (1 + uvec.y))) 			+ (zc * uvec.x);
				zr = -(xc * ((uvec.x * uvec.z) / (1 + uvec.y))) 		+ (yc * (1 - (powf(uvec.z, 2.0) / (1 + uvec.y))))  		+ (zc * uvec.z);
				yr = -(xc * uvec.x) 									- (yc * uvec.z) 										+ (zc * uvec.y);
			} else {
				xr = -(xc * (1 - (powf(uvec.x, 2.0) / (1 - uvec.y)))) 	+ (yc * ((uvec.x * uvec.z) / (1 - uvec.y))) 			+ (zc * uvec.x);
				zr =  (xc * ((uvec.x * uvec.z) / (1 - uvec.y))) 		- (yc * (1 - (powf(uvec.z, 2.0) / (1 - uvec.y))))  		+ (zc * uvec.z);
				yr = -(xc * uvec.x) 									- (yc * uvec.z) 										+ (zc * uvec.y);
			}

			Rs = 0.0;
			Rp = 0.0;
			sp = dose->spacing / 10.0 * 2.0; 

			while(ray_length >= -sp) {

				vox_ray_xyz.x = fmaf(xr, ray_length * 10.0, vox_xyz.x);
				vox_ray_xyz.y = fmaf(yr, ray_length * 10.0, vox_xyz.y);
				vox_ray_xyz.z = fmaf(zr, ray_length * 10.0, vox_xyz.z);

				dose->pointXYZtoIJK(&vox_ray_xyz, &vox_ray_ijk, beam);

				if(!dose->pointIJKWithinImage(&vox_ray_ijk)){
					break;
				}

				vox_ray_index = dose->pointIJKtoIndex(&vox_ray_ijk);

				Ti = TERMAArray[vox_ray_index];
				Di = dose->DensityArray[vox_ray_index] * sp;

				if(Di <= 0.0){
					Di = AIR_DENSITY * sp;
				}

				Rp = Rp * exp(-am * Di) + (Ti * sinf(th * M_PI / 180.0) * (Am / powf(am, 2)) * (1 - exp(-am * Di)));
				Rs = Rs * (1 - (bm * Di)) + (Ti * Di * sinf(th * M_PI / 180.0) * (Bm / bm));
				
				ray_length = ray_length - sp;

			}

			dose_value += am * Rp + bm * Rs;
		}
	}

	if(!isnan(dose_value) & (dose_value > 0.0)){
		dose->DoseArray[vox_index] = MU_CAL * dose_value;
	}

	__syncthreads();

}


void photon_dose_cuda(int gpu_id, DoseClass * h_dose, BeamClass  * h_beam){

	CUDA_CHECK(cudaSetDevice(gpu_id));

	IMRTDose d_dose(h_dose);
	IMRTBeam d_beam(h_beam);

	DevicePointer<float> DensityArray(h_dose->DensityArray, h_dose->num_voxels);
	DevicePointer<float> DoseArray(MemoryTag::Zeroed(), h_dose->num_voxels);
	DevicePointer<float> WETArray(MemoryTag::Zeroed(), h_dose->num_voxels);
	DevicePointer<float> TERMAArray(MemoryTag::Zeroed(), h_dose->num_voxels);

	d_dose.DensityArray = DensityArray.get();
	d_dose.WETArray = WETArray.get();
	d_dose.DoseArray = DoseArray.get();
	
	DevicePointer<MLCPair> MLCPairArray(h_beam->mlc, h_beam->n_mlc_pairs);

	d_beam.mlc = MLCPairArray.get();

	DevicePointer<IMRTBeam> d_beam_ptr(&d_beam);
	DevicePointer<IMRTDose> d_dose_ptr(&d_dose);

	cudaMemcpyToSymbol(g_kernel, h_kernel, 6 * 6 * sizeof(float));

	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((d_dose.img_sz.k + TILE_WIDTH - 1) / TILE_WIDTH, (d_dose.img_sz.j + TILE_WIDTH - 1) / TILE_WIDTH, (d_dose.img_sz.i + TILE_WIDTH - 1) / TILE_WIDTH);

    rayTraceKernelPhoton<<<dimGrid, dimBlock>>>(d_dose_ptr, d_beam_ptr);
    termaKernel<<<dimGrid, dimBlock>>>(d_dose_ptr, d_beam_ptr, TERMAArray);
	cccKernel<<<dimGrid, dimBlock>>>(d_dose_ptr, d_beam_ptr, TERMAArray);

	CUDA_CHECK(cudaMemcpy(h_dose->DoseArray, d_dose.DoseArray, d_dose.num_voxels * sizeof(float), cudaMemcpyDeviceToHost));

}
