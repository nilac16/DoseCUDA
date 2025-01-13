#include "./IMPTClasses.cuh"
#ifndef DOSECUDA_DEVICE_POINTER
#	define DOSECUDA_DEVICE_POINTER
#endif
#include "MemoryClasses.h"


__host__ IMPTBeam::IMPTBeam(BeamClass * h_beam) : CudaBeam(h_beam) {}


__device__ void IMPTBeam::interpolateProtonLUT(float wet, float * idd, float * sigma, size_t layer_id){

	const Layer &layer = this->layers[layer_id];
	const float *depths, *sigmas, *idds;

	depths = this->lut_depths + layer.energy_id * this->lut_len;
	sigmas = this->lut_sigmas + layer.energy_id * this->lut_len;
	idds = this->lut_idds + layer.energy_id * this->lut_len;

	int i = 0, j = this->lut_len, mid;

	do {
		mid = (i + j) / 2;
		if (wet < depths[mid]) {
			j = mid;
		} else {
			i = mid + 1;
		}
	} while (i < j);

	if (i >= this->lut_len-1){
		*idd = idds[this->lut_len-1];
		*sigma = sigmas[this->lut_len-1];
	} else if (i <= 1) {
		*idd = idds[0];
		*sigma = sigmas[0];
	} else {
		*idd = (((idds[i] - idds[i-1]) / (depths[i] - depths[i-1])) * (wet - depths[i-1])) + idds[i-1];
		*sigma = (((sigmas[i] - sigmas[i-1]) / (depths[i] - depths[i-1])) * (wet - depths[i-1])) + sigmas[i-1];
	}

}

__device__ float IMPTBeam::sigmaAir(float wet, float distance_to_source, size_t layer_id) {

	Layer &layer = this->layers[layer_id];
	float d = distance_to_source - wet + (0.7 * layer.r80);
	const float *coef = &this->divergence_params[this->dvp_len * layer.energy_id + 2];

	return fmaf(fmaf(coef[0], d, coef[1]), d, coef[2]);

}

__device__ float clamp(float x, float lo, float hi)
{
	return fmaxf(fminf(x, hi), lo);
}

__device__ float sqr(float x)
{
	return x * x;
}

__device__ void IMPTBeam::nuclearHalo(float wet, float * halo_sigma, float * halo_weight, size_t layer_id) {

	Layer &layer = this->layers[layer_id];

	wet = clamp(wet, 0.1, layer.r80 - 0.1);

	float halo_sigma_ = 2.85 + (0.0014 * layer.r80 * logf(wet + 3.0)) +
	(0.06 * wet) - (7.4e-5 * sqr(wet)) -
	((0.22 * layer.r80) / sqr(wet - layer.r80 - 5.0));

	float halo_weight_ = 0.052 * logf(1.13 + (wet / (11.2 - (0.023 * layer.r80)))) +
	(0.35 * ((0.0017 * sqr(layer.r80)) - layer.r80) / (sqr(layer.r80 + 3.0) - sqr(wet))) -
	(1.61e-9 * wet * sqr(layer.r80 + 3.0));

	*halo_sigma = fmaxf(halo_sigma_, 0.0);
	*halo_weight = clamp(halo_weight_, 0.0, 0.9);

}


__host__ IMPTDose::IMPTDose(DoseClass * h_dose) : CudaDose(h_dose) {}


__global__ void smoothRayKernel(IMPTDose * dose, IMPTBeam * beam, float * SmoothedWETArray){

	PointIJK vox_ijk;
	vox_ijk.k = threadIdx.x + (blockIdx.x * blockDim.x);
	vox_ijk.j = threadIdx.y + (blockIdx.y * blockDim.y);
	vox_ijk.i = threadIdx.z + (blockIdx.z * blockDim.z);

	if(!dose->pointIJKWithinImage(&vox_ijk)) {
		return;
	}

	int vox_index = dose->pointIJKtoIndex(&vox_ijk);

	PointXYZ vox_xyz;
	dose->pointIJKtoXYZ(&vox_ijk, &vox_xyz, beam);

	PointXYZ uvec;
	beam->unitVectorToSource(&vox_xyz, &uvec);

	float step_length = 1.0;

	PointXYZ vox_head_xyz;
	PointXYZ vox_head_xyz_conv;
	PointXYZ vox_image_xyz_conv;
	PointIJK vox_ijk_conv;
	int vox_index_conv;

	float center_wet = dose->WETArray[vox_index];
	float dr = 1.0;
	float conv_wet_sum = dose->WETArray[vox_index];
	int n_voxels = 1;

	dose->pointXYZImageToHead(&vox_xyz, &vox_head_xyz, beam);

	for (int i=0; i<6; i++){
		float sinx, cosx;

		dr = 1.0;
		sincosf((float)i * CUDART_PI_F / 3.0, &sinx, &cosx);

		while ((dr < (center_wet * 10.0)) & (dr < 10.0)){

			vox_head_xyz_conv.x = vox_head_xyz.x + (dr * cosx);
			vox_head_xyz_conv.y = vox_head_xyz.y;
			vox_head_xyz_conv.z = vox_head_xyz.z + (dr * sinx);

			dose->pointXYZHeadToImage(&vox_head_xyz_conv, &vox_image_xyz_conv, beam);
			dose->pointXYZtoIJK(&vox_image_xyz_conv, &vox_ijk_conv, beam);

			if(dose->pointIJKWithinImage(&vox_ijk_conv)){
				vox_index_conv = dose->pointIJKtoIndex(&vox_ijk_conv);
				conv_wet_sum += dose->WETArray[vox_index_conv];
				n_voxels += 1;
			}

			dr += step_length;
		}
	}

	SmoothedWETArray[vox_index] = conv_wet_sum / (float)n_voxels;

	__syncthreads();

}

__device__ float gauss(float x, float s)
{
	x /= s;
	return expf(-0.5 * x * x);
}

__global__ void pencilBeamKernel(IMPTDose * dose, IMPTBeam * beam){

	PointIJK vox_ijk;
	vox_ijk.k = threadIdx.x + (blockIdx.x * blockDim.x);
	vox_ijk.j = threadIdx.y + (blockIdx.y * blockDim.y);
	vox_ijk.i = (threadIdx.z + (blockIdx.z * blockDim.z)) / beam->n_layers;
	size_t layer_id = (threadIdx.z + (blockIdx.z * blockDim.z)) % beam->n_layers;

	if (layer_id >= beam->n_layers){
		return;
	}

	if(!dose->pointIJKWithinImage(&vox_ijk)) {
		return;
	}

	size_t vox_index = dose->pointIJKtoIndex(&vox_ijk);

	float wet = dose->WETArray[vox_index] * 10.0;

	if (wet > (1.1 * beam->layers[layer_id].r80)){
		return;
	}

	PointXYZ vox_xyz, vox_head_xyz;
	dose->pointIJKtoXYZ(&vox_ijk, &vox_xyz, beam);

	float sigma_ms, idd;
	beam->interpolateProtonLUT(wet, &idd, &sigma_ms, layer_id);

	float distance_to_source = dose->pointXYZDistanceToSource(&vox_xyz, beam);
	float sigma_total = beam->sigmaAir(wet, distance_to_source, layer_id) + sigma_ms;

	float sigma_halo, halo_weight;
	beam->nuclearHalo(wet, &sigma_halo, &halo_weight, layer_id);
	float sigma_halo_total = hypotf(sigma_total, sigma_halo);

	Layer &layer = beam->layers[layer_id];

	float primary_dose_factor = (1.0 - halo_weight) * idd / (2.0 * CUDART_PI_F * sqr(sigma_total));
	float halo_dose_factor = halo_weight * idd / (2.0 * CUDART_PI_F * sqr(sigma_halo_total));

	float total_dose = 0.0, distance_to_cax, primary_dose, halo_dose;

	dose->pointXYZImageToHead(&vox_xyz, &vox_head_xyz, beam);
	float vx = -vox_head_xyz.x;
	float vy = vox_head_xyz.z;
	float vz = -vox_head_xyz.y;
	float sx, sy, sz = (VSADX + VSADY) / 2.0;

	const int spot_end = layer.spot_start + layer.n_spots;

	for (int spot_id=layer.spot_start; spot_id < spot_end; spot_id++){

		const Spot &spot = beam->spots[spot_id];

		sx = spot.x;
		sy = spot.y;

		if ((fabsf(vx - sx) > 50.0) || (fabsf(vy - sy) > 50.0)){
			continue;
		}

		{
			float dx, dy, dz;

			dx = sy * vz - sz * (vy - sy);
			dy = sz * (vx - sx) - sx * vz;
			dz = sx * (vy - sy) - sy * (vx - sx);

			distance_to_cax = norm3df(dx, dy, dz) * rnorm3df(sx, sy, sz);
		}

		primary_dose = primary_dose_factor * gauss(distance_to_cax, sigma_total);
		if (isnan(primary_dose)) {
			continue;
		}

		halo_dose = halo_dose_factor * gauss(distance_to_cax, sigma_halo_total);
		if (isnan(halo_dose)) {
			continue;
		}

		total_dose = fmaf(spot.mu, primary_dose + halo_dose, total_dose);

	}

	atomicAdd(&dose->DoseArray[vox_index], total_dose);

	__syncthreads();

}


void proton_raytrace_cuda(int gpu_id, DoseClass * h_dose, BeamClass  * h_beam){

	CUDA_CHECK(cudaSetDevice(gpu_id));

	IMPTDose d_dose(h_dose);
	IMPTBeam d_beam(h_beam);

	DevicePointer<float> DensityArray(h_dose->DensityArray, h_dose->num_voxels);
	DevicePointer<float> WETArray(h_dose->num_voxels);
	DevicePointer<float> SmoothedWETArray(d_dose.num_voxels);

	d_dose.DensityArray = DensityArray.get();
	d_dose.WETArray = WETArray.get();

	DevicePointer<IMPTDose> d_dose_ptr(&d_dose);
	DevicePointer<IMPTBeam> d_beam_ptr(&d_beam);

	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((d_dose.img_sz.k + TILE_WIDTH - 1) / TILE_WIDTH, (d_dose.img_sz.j + TILE_WIDTH - 1) / TILE_WIDTH, (d_dose.img_sz.i + TILE_WIDTH - 1) / TILE_WIDTH);

	auto DensityTexture = Texture3D::fromHostData(h_dose->DensityArray, h_dose->img_sz, cudaFilterModeLinear);

	rayTraceKernel<<<dimGrid, dimBlock>>>(d_dose_ptr, d_beam_ptr, DensityTexture);
	smoothRayKernel<<<dimGrid, dimBlock>>>(d_dose_ptr, d_beam_ptr, SmoothedWETArray);

	CUDA_CHECK(cudaMemcpy(h_dose->WETArray, SmoothedWETArray.get(), d_dose.num_voxels * sizeof(float), cudaMemcpyDeviceToHost));

}

void proton_spot_cuda(int gpu_id, DoseClass * h_dose, BeamClass  * h_beam){

	CUDA_CHECK(cudaSetDevice(gpu_id));

	IMPTDose d_dose(h_dose);
	IMPTBeam d_beam(h_beam);

	DevicePointer<float> DoseArray(MemoryTag::Zeroed(), h_dose->num_voxels);
	DevicePointer<float> DensityArray(h_dose->DensityArray, h_dose->num_voxels);
	DevicePointer<float> WETArray(h_dose->WETArray, h_dose->num_voxels);

	d_dose.DoseArray = DoseArray.get();
	d_dose.DensityArray = DensityArray.get();
	d_dose.WETArray = WETArray.get();

	DevicePointer<Layer> LayerArray(h_beam->layers, h_beam->n_layers);
	DevicePointer<Spot> SpotArray(h_beam->spots, h_beam->n_spots);
	DevicePointer<float> DivergenceParams(h_beam->divergence_params, h_beam->dvp_len * h_beam->n_energies);
	DevicePointer<float> LUTDepths(h_beam->lut_depths, h_beam->lut_len * h_beam->n_energies);
	DevicePointer<float> LUTSigmas(h_beam->lut_sigmas, h_beam->lut_len * h_beam->n_energies);
	DevicePointer<float> LUTIDDs(h_beam->lut_idds, h_beam->lut_len * h_beam->n_energies);

	d_beam.layers = LayerArray.get();
	d_beam.spots = SpotArray.get();
	d_beam.divergence_params = DivergenceParams.get();
	d_beam.lut_depths = LUTDepths.get();
	d_beam.lut_sigmas = LUTSigmas.get();
	d_beam.lut_idds = LUTIDDs.get();

	DevicePointer<IMPTBeam> d_beam_ptr(&d_beam);
	DevicePointer<IMPTDose> d_dose_ptr(&d_dose);

	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((d_dose.img_sz.k + TILE_WIDTH - 1) / TILE_WIDTH, (d_dose.img_sz.j + TILE_WIDTH - 1) / TILE_WIDTH, ((d_dose.img_sz.i * (int)d_beam.n_layers) + TILE_WIDTH - 1) / TILE_WIDTH);

	pencilBeamKernel<<<dimGrid, dimBlock>>>(d_dose_ptr, d_beam_ptr);

	CUDA_CHECK(cudaMemcpy(h_dose->DoseArray, d_dose.DoseArray, d_dose.num_voxels * sizeof(float), cudaMemcpyDeviceToHost));

}
