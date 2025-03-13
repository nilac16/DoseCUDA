#include "./IMRTClasses.cuh"
#include "MemoryClasses.h"


__host__ __device__ float interp(float x, const float * xd, const float * yd, const int n){

	int i = lowerBound(xd, n, x);

	if (i == n) {
		return yd[i - 1];
	} else if (i == 0) {
		return yd[i];
	} else {
		auto factor = (x - xd[i - 1]) / (xd[i] - xd[i - 1]);
		return fmaf(yd[i] - yd[i - 1], factor, yd[i - 1]);
	}
}

__host__ IMRTBeam::IMRTBeam(IMRTBeam * h_beam) : CudaBeam(h_beam) {

	this->model = h_beam->model;
	this->collimator_angle = h_beam->collimator_angle;
	this->mu = h_beam->mu;
	this->sinca = h_beam->sinca;
	this->cosca = h_beam->cosca;
	this->n_mlc_pairs = h_beam->n_mlc_pairs;

}

__host__ IMRTBeam::IMRTBeam(float * iso, float gantry_angle, float couch_angle, float collimator_angle, const Model * model)
		: CudaBeam(iso, gantry_angle, couch_angle, model->primary_src_dist) {

	this->model = *model;
	this->collimator_angle = collimator_angle;

	float ca = collimator_angle * M_PI / 180.0f;
	this->sinca = sin(ca);
	this->cosca = cos(ca);
}

__device__ void IMRTBeam::pointXYZImageToHead(const PointXYZ * point_img, PointXYZ * point_head){

	float sinx, cosx;

	//table rotation - rotate about y-axis
	float xt, yt, zt;
	sinx = -this->sinta;
	cosx = this->costa;
	xt = point_img->x * cosx + point_img->z * sinx;
	yt = point_img->y;
	zt = -point_img->x * sinx + point_img->z * cosx;

	//gantry rotation - rotate about z-axis
	float xg, yg, zg;
	sinx = -this->singa;
	cosx = this->cosga;
	xg = xt * cosx - yt * sinx;
	yg = xt * sinx + yt * cosx;
	zg = zt;

	//collimator rotation = rotate about y-axis
	float xc, yc, zc;
	sinx = -this->sinca;
	cosx = this->cosca;
	xc  = xg * cosx + zg * sinx;
	yc = yg;
	zc = -xg * sinx + zg * cosx;


	//swap final coordinates to match DICOM nozzle coordinate system
	//for an AP beam:
	//	beam travels in negative z direction
	//	positive x is to the patient's left
	//	positive y is to the patient's superior
	point_head->x = -xc;
	point_head->y = zc;
	point_head->z = yc;

}

__device__ void IMRTBeam::pointXYZHeadToImage(const PointXYZ * point_head, PointXYZ * point_img){

	float sinx, cosx;

	//convert back to DICOM patient LPS coordinates
	float xz = -point_head->x;
	float yz = point_head->z;
	float zz = point_head->y;

	//collimator rotation = rotate about y-axis (again, negative direction)
	float xc, yc, zc;
	sinx = this->sinca;
	cosx = this->cosca;
	xc  = xz * cosx + zz * sinx;
	yc = yz;
	zc = -xz * sinx + zz * cosx;

	//gantry rotation - rotate about z-axis (negative direction)
	float xg, yg, zg;
	sinx = this->singa;
	cosx = this->cosga;
	xg = xc * cosx - yc * sinx;
	yg = xc * sinx + yc * cosx;
	zg = zc;

	//table rotation - rotate about y-axis (negative direction)
	float xt, yt, zt;
	sinx = this->sinta;
	cosx = this->costa;
	xt = xg * cosx + zg * sinx;
	yt = yg;
	zt = -xg * sinx + zg * cosx;

	point_img->x = xt;
	point_img->y = yt;
	point_img->z = zt;

}

__device__ float IMRTBeam::headTransmission(const PointXYZ* point_xyz, const float iso_to_source, const float source_sigma){

	const float mlc_scale = (model.primary_src_dist - model.mlc_distance) / model.primary_src_dist;
	const float divergence_scale = (iso_to_source - point_xyz->z) / (model.mlc_distance - point_xyz->z);
	const float invSqrt2_x = 1.0f / (source_sigma * sqrtf(2.f));
    const float invSqrt2_y = 1.0f / (source_sigma * sqrtf(2.f));
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

	const float distance_to_source = model.primary_src_dist - point_xyz->z;
	const float distance_to_cax = sqrtf(powf(point_xyz->x, 2.0) + powf(point_xyz->y, 2.0)) * distance_to_source / model.primary_src_dist;

	int i = lowerBound(this->model.profile_radius, this->model.n_profile_points, distance_to_cax);

	if (i == this->model.n_profile_points) {
		*off_axis_factor = this->model.profile_intensities[i - 1];
		*off_axis_softening = this->model.profile_softening[i - 1];
	} else if (i == 0) {
		*off_axis_factor = this->model.profile_intensities[i];
		*off_axis_softening = this->model.profile_softening[i];
	} else {
		auto mult = (distance_to_cax - this->model.profile_radius[i - 1]) / (this->model.profile_radius[i] - this->model.profile_radius[i - 1]);
		*off_axis_factor = fmaf(this->model.profile_intensities[i] - this->model.profile_intensities[i - 1], mult, this->model.profile_intensities[i - 1]);
		*off_axis_softening = fmaf(this->model.profile_softening[i] - this->model.profile_softening[i - 1], mult, this->model.profile_softening[i - 1]);
	}

}


__host__ IMRTDose::IMRTDose(CudaDose * h_dose) : CudaDose(h_dose) {}


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
	beam->pointXYZImageToHead(&vox_xyz, &vox_head_xyz);

	float distance_to_primary_source = beam->model.primary_src_dist - vox_head_xyz.z;
	float distance_to_scatter_source = beam->model.scatter_src_dist - vox_head_xyz.z;
	float off_axis_factor, off_axis_softening;
	beam->offAxisFactors(&vox_head_xyz, &off_axis_factor, &off_axis_softening);
	float primary_transmission = beam->headTransmission(&vox_head_xyz, beam->model.primary_src_dist, beam->model.primary_src_size);
	float scatter_transmission = beam->headTransmission(&vox_head_xyz, beam->model.scatter_src_dist, beam->model.scatter_src_size);
	float wet = dose->WETArray[vox_index];
	float terma = 0.f;
	float electron = 0.f;
	float transmission_ratio = fminf(1.00, scatter_transmission / primary_transmission);

	for(int i = 0; i < beam->model.n_spectral_energies; i++){
		terma += (1.0 - transmission_ratio) * (beam->model.spectrum_primary_weights[i] * expf(-beam->model.spectrum_attenuation_coefficients[i] * wet * off_axis_softening) * powf(beam->model.primary_src_dist / distance_to_primary_source, 2.0)) + 
					transmission_ratio * beam->model.spectrum_scatter_weights[i] * expf(-beam->model.spectrum_attenuation_coefficients[i] * wet * off_axis_softening) * (beam->model.scatter_src_dist / distance_to_scatter_source);
	}

	electron = fmaxf(0.0, (expf(-beam->model.electron_attenuation * wet) - expf(-beam->model.electron_attenuation * beam->model.electron_fitted_dmax)) / (1.0 - expf(-beam->model.electron_attenuation * beam->model.electron_fitted_dmax)));

	TERMAArray[vox_index] = off_axis_factor * ((1.0 - beam->model.scatter_src_weight) * primary_transmission * terma + beam->model.scatter_src_weight * scatter_transmission * terma);

	ElectronArray[vox_index] = beam->model.electron_src_weight * (0.4 + (0.3 * transmission_ratio)) * electron * primary_transmission;

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
	beam->pointXYZImageToHead(&vox_img_xyz, &vox_head_xyz);

	float dose_value = 0.0;
	float sp = dose->spacing / 10.0; //cm
	
	for(int i = 0; i < 6; i++){

		// float th = g_kernel[0][i] * M_PI / 180.0;
		// float Am = g_kernel[1][i];
		// float am = g_kernel[2][i];
		// float Bm = g_kernel[3][i];
		// float bm = g_kernel[4][i];
		// float ray_length_init = g_kernel[5][i];

		float th = beam->model.kernel[i] * M_PI / 180.0;
		float Am = beam->model.kernel[i + 6];
		float am = beam->model.kernel[i + 12];
		float Bm = beam->model.kernel[i + 18];
		float bm = beam->model.kernel[i + 24];
		float ray_length_init = beam->model.kernel[i + 30];

		for(int j = 0; j < 12; j++){

			float phi = (float)j * 30.0 * M_PI / 180.0;
			float xr = sinf(th) * cosf(phi);
			float yr = sinf(th) * sinf(phi);
			float zr = cosf(th);

			float Rs = 0.0, Rp = 0.0, Ti = 0.0;
			float Di = AIR_DENSITY * sp;
			float ray_length = ray_length_init;

			while(ray_length >= 0) {

				PointXYZ ray_head_xyz;
				ray_head_xyz.x = fmaf(xr, ray_length * 10.0, vox_head_xyz.x);
				ray_head_xyz.y = fmaf(yr, ray_length * 10.0, vox_head_xyz.y);
				ray_head_xyz.z = fmaf(zr, ray_length * 10.0, vox_head_xyz.z);

				PointXYZ ray_img_xyz;
				beam->pointXYZHeadToImage(&ray_head_xyz, &ray_img_xyz);

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
		dose->DoseArray[vox_index] = beam->model.mu_cal * dose_value * beam->mu;
	}

	__syncthreads();

}


void photon_dose_cuda(int gpu_id, IMRTDose * h_dose, IMRTBeam * h_beam){

	CUDA_CHECK(cudaSetDevice(gpu_id));

	IMRTDose d_dose(h_dose);
	IMRTBeam d_beam(h_beam);

	DevicePointer<float> DoseArray(MemoryTag::Zeroed(), h_dose->num_voxels);
	DevicePointer<float> WETArray(MemoryTag::Zeroed(), h_dose->num_voxels);
	DevicePointer<float> TERMAArray(MemoryTag::Zeroed(), h_dose->num_voxels);
	DevicePointer<float> ElectronArray(MemoryTag::Zeroed(), h_dose->num_voxels);

	d_dose.WETArray = WETArray.get();
	d_dose.DoseArray = DoseArray.get();
	
	DevicePointer<MLCPair> MLCPairArray(h_beam->mlc, h_beam->n_mlc_pairs);
	DevicePointer<float> d_profile_radius(h_beam->model.profile_radius, h_beam->model.n_profile_points);
	DevicePointer<float> d_profile_intensities(h_beam->model.profile_intensities, h_beam->model.n_profile_points);
	DevicePointer<float> d_profile_softening(h_beam->model.profile_softening, h_beam->model.n_profile_points);
	DevicePointer<float> d_spectrum_attenuation_coefficients(h_beam->model.spectrum_attenuation_coefficients, h_beam->model.n_spectral_energies);
	DevicePointer<float> d_spectrum_primary_weights(h_beam->model.spectrum_primary_weights, h_beam->model.n_spectral_energies);
	DevicePointer<float> d_spectrum_scatter_weights(h_beam->model.spectrum_scatter_weights, h_beam->model.n_spectral_energies);
	DevicePointer<float> d_kernel(h_beam->model.kernel, 6 * 6);

	d_beam.mlc = MLCPairArray.get();
	d_beam.model.profile_radius = d_profile_radius.get();
	d_beam.model.profile_intensities = d_profile_intensities.get();
	d_beam.model.profile_softening = d_profile_softening.get();
	d_beam.model.spectrum_attenuation_coefficients = d_spectrum_attenuation_coefficients.get();
	d_beam.model.spectrum_primary_weights = d_spectrum_primary_weights.get();
	d_beam.model.spectrum_scatter_weights = d_spectrum_scatter_weights.get();
	d_beam.model.kernel = d_kernel.get();

	// CUDA_CHECK(cudaMemcpyToSymbol(g_kernel, h_kernel, 6 * 6 * sizeof(float)));

	DevicePointer<IMRTBeam> d_beam_ptr(&d_beam);
	DevicePointer<IMRTDose> d_dose_ptr(&d_dose);

	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((d_dose.img_sz.k + TILE_WIDTH - 1) / TILE_WIDTH, (d_dose.img_sz.j + TILE_WIDTH - 1) / TILE_WIDTH, (d_dose.img_sz.i + TILE_WIDTH - 1) / TILE_WIDTH);

	auto DensityTexture = Texture3D::fromHostData(h_dose->DensityArray, h_dose->img_sz, cudaFilterModeLinear);

    rayTraceKernel<<<dimGrid, dimBlock>>>(d_dose_ptr, d_beam_ptr, DensityTexture);
    termaKernel<<<dimGrid, dimBlock>>>(d_dose_ptr, d_beam_ptr, TERMAArray, ElectronArray);

	auto TERMATexture = Texture3D::fromDeviceData(TERMAArray, h_dose->img_sz, cudaFilterModeLinear);

	cccKernel<<<dimGrid, dimBlock>>>(d_dose_ptr, d_beam_ptr, TERMATexture, DensityTexture, ElectronArray);

	CUDA_CHECK(cudaMemcpy(h_dose->DoseArray, d_dose.DoseArray, d_dose.num_voxels * sizeof(float), cudaMemcpyDeviceToHost));

}
