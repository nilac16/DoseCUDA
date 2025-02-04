#include "./IMRTClasses.cuh"
#include "MemoryClasses.h"


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

const float h_profile[] = 						{1.000, 1.009, 1.020, 1.031, 1.039, 1.05, 1.06, 1.07, 1.06, 0.960,
												0.940, 0.920, 0.900, 0.875, 0.780, 0.692, 0.050, 0.010, 0.001, 0.001};

const float h_soften[] = 						{1.000, 1.040, 1.080, 1.120, 1.160, 1.200, 1.240, 1.300, 1.320, 1.360,
												1.420, 1.440, 1.460, 1.480, 1.490, 1.500, 1.520, 1.560, 1.590, 1.600};

const float h_radius[] = 						{0.000, 20.00, 40.00, 60.00, 80.00, 100.0, 120.0, 150.0, 160.0, 180.0,
												210.0, 220.0, 230.0, 240.0, 245.0, 250.0, 260.0, 280.0, 295.0, 300.0};


__constant__ float g_kernel[6][6]; //TERMA to dose kernel
__constant__ float g_attenuation_coefficients[12]; //attenuation coefficients
__constant__ float g_energy_fluence[12]; //energy fluence
__constant__ float g_scatter_energy_fluence[12]; //scatter energy fluence
//__constant__ float g_profile[20]; //off axis intensity
//__constant__ float g_soften[20]; //off axis softening factor
//__constant__ float g_radius[20]; //radius of profile/softening points


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

	this->off_axis_radii = h_beam->off_axis_radii;
	this->off_axis_factors = h_beam->off_axis_factors;
	this->off_axis_softening = h_beam->off_axis_softening;
	this->off_axis_len = h_beam->off_axis_len;
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

__device__ float IMRTBeam::headTransmission(const PointXYZ* point_xyz, const float iso_to_source, const float xSigma, const float ySigma){

	const float mlc_scale = (model.primary_src_dist - model.mlc_distance) / model.primary_src_dist;
	const float divergence_scale = (iso_to_source - point_xyz->z) / (model.mlc_distance - point_xyz->z);
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

	const float distance_to_source = model.primary_src_dist - point_xyz->z;
	const float distance_to_cax = sqrtf(powf(point_xyz->x, 2.0) + powf(point_xyz->y, 2.0)) * distance_to_source / model.primary_src_dist;

	int i = lowerBound(this->off_axis_radii, this->off_axis_len, distance_to_cax);

	if (i == this->off_axis_len) {
		*off_axis_factor = this->off_axis_factors[i - 1];
		*off_axis_softening = this->off_axis_softening[i - 1];
	} else if (i == 0) {
		*off_axis_factor = this->off_axis_factors[i];
		*off_axis_softening = this->off_axis_softening[i];
	} else {
		auto mult = (distance_to_cax - this->off_axis_radii[i - 1]) / (this->off_axis_radii[i] - this->off_axis_radii[i - 1]);
		*off_axis_factor = fmaf(this->off_axis_factors[i] - this->off_axis_factors[i - 1], mult, this->off_axis_factors[i - 1]);
		*off_axis_softening = fmaf(this->off_axis_softening[i] - this->off_axis_softening[i - 1], mult, this->off_axis_softening[i - 1]);
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
	float primary_transmission = beam->headTransmission(&vox_head_xyz, beam->model.primary_src_dist, 1.0, 1.0);
	float scatter_transmission = beam->headTransmission(&vox_head_xyz, beam->model.scatter_src_dist, 30.0, 30.0);
	float wet = dose->WETArray[vox_index];
	float terma = 0.f;
	float electron = 0.f;
	float transmission_ratio = fminf(1.00, scatter_transmission / primary_transmission);

	for(int i = 0; i < 12; i++){
		terma += (1.0 - transmission_ratio) * (g_energy_fluence[i] * expf(-g_attenuation_coefficients[i] * wet * off_axis_softening) * powf(beam->model.primary_src_dist / distance_to_primary_source, 2.0)) + 
					transmission_ratio * g_scatter_energy_fluence[i] * expf(-g_attenuation_coefficients[i] * wet * off_axis_softening) * (beam->model.scatter_src_dist / distance_to_scatter_source);
	}

	electron = fmaxf(0.0, (expf(-beam->model.electron_mass_attenuation * wet) - expf(-beam->model.electron_mass_attenuation * 1.5)) / (1.0 - expf(-beam->model.electron_mass_attenuation * 1.5)));

	TERMAArray[vox_index] = off_axis_factor * ((1.0 - beam->model.scatter_src_weight) * primary_transmission * terma + beam->model.scatter_src_weight * scatter_transmission * terma);

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

	// if (TERMATexture.sample(tex_img_xyz) <= 0.01){
	// 	dose->DoseArray[vox_index] = 0.0;
	// 	return;
	// }

	PointXYZ vox_head_xyz;
	beam->pointXYZImageToHead(&vox_img_xyz, &vox_head_xyz);

	float dose_value = 0.0;
	float sp = dose->spacing / 10.0; //cm
	
	for(int i = 0; i < 6; i++){

		float th = g_kernel[0][i] * M_PI / 180.0;
		float Am = g_kernel[1][i];
		float am = g_kernel[2][i];
		float Bm = g_kernel[3][i];
		float bm = g_kernel[4][i];
		float ray_length_init = g_kernel[5][i];

		for(int j = 0; j < 12; j++){

			float phi = (float)j * 30.0 * M_PI / 180.0;
			float xr = sinf(th) * cosf(phi);
			float yr = sinf(th) * sinf(phi);
			float zr = cosf(th);

			float Rs = 0.0, Rp = 0.0, Ti = 0.0;
			float Di = beam->model.air_density * sp;
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
					Di = beam->model.air_density * sp;
				}

				if(Di <= 0.0){
					Di = beam->model.air_density * sp;
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

	DevicePointer<float> OffAxisFactors(h_profile, 20);
	DevicePointer<float> OffAxisSoftening(h_soften, 20);
	DevicePointer<float> OffAxisRadii(h_radius, 20);

	d_beam.mlc = MLCPairArray.get();

	d_beam.off_axis_radii = OffAxisRadii.get();
	d_beam.off_axis_factors = OffAxisFactors.get();
	d_beam.off_axis_softening = OffAxisSoftening.get();
	d_beam.off_axis_len = 20;

	DevicePointer<IMRTBeam> d_beam_ptr(&d_beam);
	DevicePointer<IMRTDose> d_dose_ptr(&d_dose);

	CUDA_CHECK(cudaMemcpyToSymbol(g_kernel, h_kernel, 6 * 6 * sizeof(float)));
	CUDA_CHECK(cudaMemcpyToSymbol(g_attenuation_coefficients, h_attenuation_coefficients, 12 * sizeof(float)));
	CUDA_CHECK(cudaMemcpyToSymbol(g_energy_fluence, h_energy_fluence, 12 * sizeof(float)));
	CUDA_CHECK(cudaMemcpyToSymbol(g_scatter_energy_fluence, h_scatter_energy_fluence, 12 * sizeof(float)));
	//CUDA_CHECK(cudaMemcpyToSymbol(g_profile, h_profile, 20 * sizeof(float)));
	//CUDA_CHECK(cudaMemcpyToSymbol(g_soften, h_soften, 20 * sizeof(float)));
	//CUDA_CHECK(cudaMemcpyToSymbol(g_radius, h_radius, 20 * sizeof(float)));

	dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((d_dose.img_sz.k + TILE_WIDTH - 1) / TILE_WIDTH, (d_dose.img_sz.j + TILE_WIDTH - 1) / TILE_WIDTH, (d_dose.img_sz.i + TILE_WIDTH - 1) / TILE_WIDTH);

	auto DensityTexture = Texture3D::fromHostData(h_dose->DensityArray, h_dose->img_sz, cudaFilterModeLinear);

    rayTraceKernel<<<dimGrid, dimBlock>>>(d_dose_ptr, d_beam_ptr, DensityTexture);
    termaKernel<<<dimGrid, dimBlock>>>(d_dose_ptr, d_beam_ptr, TERMAArray, ElectronArray);

	auto TERMATexture = Texture3D::fromDeviceData(TERMAArray, h_dose->img_sz, cudaFilterModeLinear);

	cccKernel<<<dimGrid, dimBlock>>>(d_dose_ptr, d_beam_ptr, TERMATexture, DensityTexture, ElectronArray);

	CUDA_CHECK(cudaMemcpy(h_dose->DoseArray, d_dose.DoseArray, d_dose.num_voxels * sizeof(float), cudaMemcpyDeviceToHost));

}
