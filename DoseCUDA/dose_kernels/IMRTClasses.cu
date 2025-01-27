#include "./IMRTClasses.cuh"
#ifndef DOSECUDA_DEVICE_POINTER
#	define DOSECUDA_DEVICE_POINTER
#endif
#include "MemoryClasses.h"

__host__ __device__ float interp(float x, const float * xd, const float * yd, const int n){

	int i = 0;
	while ((xd[i] < x) && (i < n)){
		i++;
	}

	float y = 1.00;
	if(i >= (n - 1)){
		y = yd[n - 1];
	}else if(i <= 0){
		y = yd[0];
	} else {
		y = yd[i - 1] + ((yd[i] - yd[i - 1]) * (x - xd[i - 1]) / (xd[i] - xd[i - 1]));
	}

	return y;
}

__host__ IMRTBeam::IMRTBeam(BeamClass * h_beam) : CudaBeam(h_beam) {}

__device__ float IMRTBeam::headTransmission(const PointXYZ* point_xyz, const float iso_to_source, const float xSigma, const float ySigma){

	const float mlc_scale = (PRIMARY_SOURCE_DISTANCE - MLC_DISTANCE) / PRIMARY_SOURCE_DISTANCE;
	const float divergence_scale = (iso_to_source - point_xyz->z) / (MLC_DISTANCE - point_xyz->z);
	const float invSqrt2_x = 1.0f / (xSigma * sqrtf(2.f));
    const float invSqrt2_y = 1.0f / (ySigma * sqrtf(2.f));
	float transmission = 0.0f;

    for (int i = 0; i < this->n_mlc_pairs; ++i) {

        const float yBottom = (this->mlc[i].y_offset - 0.5f * this->mlc[i].y_width) * mlc_scale;
        const float yTop    = (this->mlc[i].y_offset + 0.5f * this->mlc[i].y_width) * mlc_scale;

		const float xLeft  = (this->mlc[i].x1 * mlc_scale);
		const float xRight = (this->mlc[i].x2 * mlc_scale);

		const float tipMLC1 = ((xLeft - point_xyz->x) * divergence_scale) + point_xyz->x;
		const float tipMLC2 = ((xRight - point_xyz->x) * divergence_scale) + point_xyz->x;

		const float edgeMLC1 = ((yBottom - point_xyz->y) * divergence_scale) + point_xyz->y;
		const float edgeMLC2 = ((yTop - point_xyz->y) * divergence_scale) + point_xyz->y;

		const float exposedSourceX = 0.5f * (erff(tipMLC2 * invSqrt2_x) - erff(tipMLC1 * invSqrt2_x));
		const float exposedSourceY = 0.5f * (erff(edgeMLC2 * invSqrt2_y) - erff(edgeMLC1 * invSqrt2_y));

		transmission += exposedSourceX * exposedSourceY;
    }

    return transmission;

}

__device__ void IMRTBeam::offAxisFactors(const PointXYZ * point_xyz, float * off_axis_factor, float * off_axis_softening){

	const float distance_to_source = PRIMARY_SOURCE_DISTANCE - point_xyz->z;
	const float distance_to_cax = sqrtf(powf(point_xyz->x, 2.0) + powf(point_xyz->y, 2.0)) * distance_to_source / PRIMARY_SOURCE_DISTANCE;
	*off_axis_factor = interp(distance_to_cax, g_radius, g_profile, 20);
	*off_axis_softening = interp(distance_to_cax, g_radius, g_soften, 20);

}


__host__ IMRTDose::IMRTDose(DoseClass * h_dose) : CudaDose(h_dose) {}


__global__ void termaKernel(IMRTDose * dose, IMRTBeam * beam, float * TERMAArray, float * ElectronArray){

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

	float distance_to_primary_source = PRIMARY_SOURCE_DISTANCE - vox_head_xyz.z;
	float distance_to_scatter_source = SCATTER_SOURCE_DISTANCE - vox_head_xyz.z;
	float off_axis_factor, off_axis_softening;
	beam->offAxisFactors(&vox_head_xyz, &off_axis_factor, &off_axis_softening);
	float primary_transmission = beam->headTransmission(&vox_head_xyz, PRIMARY_SOURCE_DISTANCE, 1.0, 1.0);
	float scatter_transmission = beam->headTransmission(&vox_head_xyz, SCATTER_SOURCE_DISTANCE, 30.0, 30.0);
	float wet = dose->WETArray[vox_index];
	float terma = 0.f;
	float electron = 0.f;
	float transmission_ratio = fminf(1.00, scatter_transmission / primary_transmission);

	for(int i = 0; i < 12; i++){
		terma += (1.0 - transmission_ratio) * (g_energy_fluence[i] * expf(-g_attenuation_coefficients[i] * wet) * powf(PRIMARY_SOURCE_DISTANCE / distance_to_primary_source, 2.0)) + 
					transmission_ratio * g_scatter_energy_fluence[i] * expf(-g_attenuation_coefficients[i] * wet) * (SCATTER_SOURCE_DISTANCE / distance_to_scatter_source);
	}

	electron = fmaxf(0.0, (expf(-ELECTRON_MASS_ATTENUATION * wet) - expf(-ELECTRON_MASS_ATTENUATION * 1.5)) / (1.0 - expf(-ELECTRON_MASS_ATTENUATION * 1.5)));

	TERMAArray[vox_index] = off_axis_factor * ((1.0 - SCATTER_SOURCE_WEIGHT) * primary_transmission * terma + SCATTER_SOURCE_WEIGHT * scatter_transmission * terma);

	ElectronArray[vox_index] = 2.0 * (0.4 + (0.3 * transmission_ratio)) * electron * primary_transmission;

    __syncthreads();

}

__global__ void cccKernel(IMRTDose * dose, IMRTBeam * beam, Texture3D TERMATexture, Texture3D DensityTexture, float * ElectronArray){

	PointIJK vox_ijk;
	vox_ijk.k = threadIdx.x + (blockIdx.x * blockDim.x);
	vox_ijk.j = threadIdx.y + (blockIdx.y * blockDim.y);
	vox_ijk.i = threadIdx.z + (blockIdx.z * blockDim.z);

	if(!dose->pointIJKWithinImage(&vox_ijk)) {
		return;
	}

	size_t vox_index = dose->pointIJKtoIndex(&vox_ijk);

	PointXYZ vox_img_xyz;
	dose->pointIJKtoXYZ(&vox_ijk, &vox_img_xyz, beam);

	PointXYZ tex_img_xyz;
	dose->pointXYZtoTextureXYZ(&vox_img_xyz, &tex_img_xyz, beam);

	if (TERMATexture.sample(tex_img_xyz) <= 0.01){
		dose->DoseArray[vox_index] = 0.0;
		return;
	}

	PointXYZ vox_head_xyz;
	dose->pointXYZImageToHead(&vox_img_xyz, &vox_head_xyz, beam);

	float dose_value = 0.0;
	float sp = dose->spacing / 10.0; //cm
	
	for(int i = 0; i < 6; i++){

		float th = g_kernel[0][i] * M_PI / 180.0;
		float Am = g_kernel[1][i];
		float am = g_kernel[2][i];
		float Bm = g_kernel[3][i];
		float bm = g_kernel[4][i];

		for(int j = 0; j < 12; j++){

			float phi = (float)j * 30.0 * M_PI / 180.0;
			float xr = sinf(th) * cosf(phi);
			float yr = sinf(th) * sinf(phi);
			float zr = cosf(th);

			float Rs = 0.0, Rp = 0.0, Ti = 0.0;
			float Di = AIR_DENSITY * sp;
			float ray_length = g_kernel[5][i];

			while(ray_length >= 0) {

				PointXYZ ray_head_xyz;
				ray_head_xyz.x = fmaf(xr, ray_length * 10.0, vox_head_xyz.x);
				ray_head_xyz.y = fmaf(yr, ray_length * 10.0, vox_head_xyz.y);
				ray_head_xyz.z = fmaf(zr, ray_length * 10.0, vox_head_xyz.z);

				PointXYZ ray_img_xyz;
				dose->pointXYZHeadToImage(&ray_head_xyz, &ray_img_xyz, beam);

				dose->pointXYZtoTextureXYZ(&ray_img_xyz, &tex_img_xyz, beam);
				if (dose->textureXYZWithinImage(&tex_img_xyz)) {
					Ti = TERMATexture.sample(tex_img_xyz);
					Di = DensityTexture.sample(tex_img_xyz) * sp;
				} else {
					Ti = 0.0;
					Di = AIR_DENSITY * sp;
				}

				if(Di <= 0.0){
					Di = AIR_DENSITY * sp;
				}

				Rp = Rp * exp(-am * Di) + (Ti * sinf(th) * (Am / powf(am, 2)) * (1 - exp(-am * Di)));
				Rs = Rs * (1 - (bm * Di)) + (Ti * Di * sinf(th) * (Bm / bm));
				
				ray_length = ray_length - sp;

			}

			dose_value += am * Rp + bm * Rs;

		}
	}

	dose_value += ElectronArray[vox_index];

	if(!isnan(dose_value) && (dose_value >= 0.0)){
		dose->DoseArray[vox_index] = MU_CAL * dose_value * beam->mu;
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
	DevicePointer<float> ElectronArray(MemoryTag::Zeroed(), h_dose->num_voxels);

	d_dose.DensityArray = DensityArray.get();
	d_dose.WETArray = WETArray.get();
	d_dose.DoseArray = DoseArray.get();
	
	DevicePointer<MLCPair> MLCPairArray(h_beam->mlc, h_beam->n_mlc_pairs);

	d_beam.mlc = MLCPairArray.get();

	DevicePointer<IMRTBeam> d_beam_ptr(&d_beam);
	DevicePointer<IMRTDose> d_dose_ptr(&d_dose);

	cudaMemcpyToSymbol(g_kernel, h_kernel, 6 * 6 * sizeof(float));
	cudaMemcpyToSymbol(g_attenuation_coefficients, h_attenuation_coefficients, 12 * sizeof(float));
	cudaMemcpyToSymbol(g_energy_fluence, h_energy_fluence, 12 * sizeof(float));
	cudaMemcpyToSymbol(g_scatter_energy_fluence, h_scatter_energy_fluence, 12 * sizeof(float));
	cudaMemcpyToSymbol(g_profile, h_profile, 20 * sizeof(float));
	cudaMemcpyToSymbol(g_soften, h_soften, 20 * sizeof(float));
	cudaMemcpyToSymbol(g_radius, h_radius, 20 * sizeof(float));

	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((d_dose.img_sz.k + TILE_WIDTH - 1) / TILE_WIDTH, (d_dose.img_sz.j + TILE_WIDTH - 1) / TILE_WIDTH, (d_dose.img_sz.i + TILE_WIDTH - 1) / TILE_WIDTH);

	auto DensityTexture = Texture3D::fromHostData(h_dose->DensityArray, h_dose->img_sz, cudaFilterModeLinear);

    rayTraceKernel<<<dimGrid, dimBlock>>>(d_dose_ptr, d_beam_ptr, DensityTexture);
    termaKernel<<<dimGrid, dimBlock>>>(d_dose_ptr, d_beam_ptr, TERMAArray, ElectronArray);

	auto TERMATexture = Texture3D::fromDeviceData(TERMAArray, h_dose->img_sz, cudaFilterModeLinear);

	cccKernel<<<dimGrid, dimBlock>>>(d_dose_ptr, d_beam_ptr, TERMATexture, DensityTexture, ElectronArray);

	CUDA_CHECK(cudaMemcpy(h_dose->DoseArray, d_dose.DoseArray, d_dose.num_voxels * sizeof(float), cudaMemcpyDeviceToHost));

}
