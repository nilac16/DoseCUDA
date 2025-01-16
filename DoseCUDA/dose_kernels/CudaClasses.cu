#include "./CudaClasses.cuh"


__host__ CudaBeam::CudaBeam(BeamClass * h_beam) : BeamClass(h_beam) {}

__device__ void CudaBeam::unitVectorToSource(const PointXYZ * point_xyz, PointXYZ * uvec) {

	float dx = this->src.x - point_xyz->x;
	float dy = this->src.y - point_xyz->y;
	float dz = this->src.z - point_xyz->z;

	float r = rnorm3df(dx, dy, dz);

	uvec->x = dx * r;
	uvec->y = dy * r;
	uvec->z = dz * r;
}

__device__ float CudaBeam::distanceToSource(const PointXYZ * point_xyz){
	
	float dx = this->src.x - point_xyz->x;
	float dy = this->src.y - point_xyz->y;
	float dz = this->src.z - point_xyz->z;

	float r = norm3df(dx, dy, dz);

	return r;

}

__host__ CudaDose::CudaDose(DoseClass * h_dose) : DoseClass(h_dose) {}

__device__ void CudaDose::pointIJKtoXYZ(const PointIJK * point_ijk, PointXYZ * point_xyz, BeamClass * beam) {

	point_xyz->x = (float)point_ijk->i * this->spacing - beam->iso.x;
	point_xyz->y = (float)point_ijk->j * this->spacing - beam->iso.y;
	point_xyz->z = (float)point_ijk->k * this->spacing - beam->iso.z;

}

/** sincos but with a value theoretically supplied to arctangent */
__device__ void sincos_from_atan(float y, float x, float *sptr, float *cptr) {

	float slope = y / x;

	*cptr = rsqrtf(fmaf(slope, slope, 1.0));
	*sptr = slope * *cptr;
}

__device__ void CudaDose::pointXYZImageToHead(const PointXYZ * point_img, PointXYZ * point_head, BeamClass * beam){

	float sinx, cosx;

	//table rotation - rotate about y-axis
	float xt, yt, zt;
	sinx = -beam->sinta;
	cosx = beam->costa;
	xt = point_img->x * cosx + point_img->z * sinx;
	yt = point_img->y;
	zt = -point_img->x * sinx + point_img->z * cosx;

	//gantry rotation - rotate about z-axis
	float xg, yg, zg;
	sinx = -beam->singa;
	cosx = beam->cosga;
	xg = xt * cosx - yt * sinx;
	yg = xt * sinx + yt * cosx;
	zg = zt;

	//collimator rotation = rotate about y-axis
	float xc, yc, zc;
	sinx = -beam->sinca;
	cosx = beam->cosca;
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

__device__ void CudaDose::pointXYZHeadToImage(const PointXYZ * point_head, PointXYZ * point_img, BeamClass * beam){

	float sinx, cosx;

	//convert back to DICOM patient LPS coordinates
	float xz = -point_head->x;
	float yz = point_head->z;
	float zz = point_head->y;

	//collimator rotation = rotate about y-axis (again, negative direction)
	float xc, yc, zc;
	sinx = beam->sinca;
	cosx = beam->cosca;
	xc  = xz * cosx + zz * sinx;
	yc = yz;
	zc = -xz * sinx + zz * cosx;

	//gantry rotation - rotate about z-axis (negative direction)
	float xg, yg, zg;
	sinx = beam->singa;
	cosx = beam->cosga;
	xg = xc * cosx - yc * sinx;
	yg = xc * sinx + yc * cosx;
	zg = zc;

	//table rotation - rotate about y-axis (negative direction)
	float xt, yt, zt;
	sinx = beam->sinta;
	cosx = beam->costa;
	xt = xg * cosx + zg * sinx;
	yt = yg;
	zt = -xg * sinx + zg * cosx;

	point_img->x = xt;
	point_img->y = yt;
	point_img->z = zt;

}

__device__ void CudaDose::pointXYZClosestCAXPoint(const PointXYZ * point_xyz, PointXYZ * point_cax, BeamClass * beam){

	float d1x = beam->iso.x - beam->src.x;
	float d1y = beam->iso.y - beam->src.y;
	float d1z = beam->iso.z - beam->src.z;

	float d2x = point_xyz->x - beam->src.x;
	float d2y = point_xyz->y - beam->src.y;
	float d2z = point_xyz->z - beam->src.z;

	float t = (d1x * d2x + d1y * d2y + d1z * d2z) / (d1x * d1x + d1y * d1y + d1z * d1z);

	point_cax->x = fmaf(t, d1x, beam->src.x);
	point_cax->y = fmaf(t, d1y, beam->src.y);
	point_cax->z = fmaf(t, d1z, beam->src.z);

}

__device__ float CudaDose::pointXYZDistanceToCAX(const PointXYZ * point_head_xyz){

	return hypotf(point_head_xyz->x, point_head_xyz->z);

}

__device__ float CudaDose::pointXYZDistanceToSource(const PointXYZ * point_img_xyz, BeamClass * beam){

	float dx = beam->src.x - point_img_xyz->x;
	float dy = beam->src.y - point_img_xyz->y;
	float dz = beam->src.z - point_img_xyz->z;

	return norm3df(dx, dy, dz);

}

__global__ void rayTraceKernel(CudaDose * dose, CudaBeam * beam, Texture3D DensityTexture){

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

	PointXYZ vox_ray_xyz, tex_xyz;
	PointIJK vox_ray_ijk;

	int vox_ray_index = 0;
    float ray_length = 0.0;
    float wet_sum = -0.05;
    float density = 0.0;
	const float step_length = 1.0;

    while(true){

		vox_ray_xyz.x = fmaf(uvec.x, ray_length, vox_xyz.x);
		vox_ray_xyz.y = fmaf(uvec.y, ray_length, vox_xyz.y);
		vox_ray_xyz.z = fmaf(uvec.z, ray_length, vox_xyz.z);

		dose->pointXYZtoTextureXYZ(&vox_ray_xyz, &tex_xyz, beam);
		if (!dose->textureXYZWithinImage(&tex_xyz)) {
			break;
		}
		density = DensityTexture.sample(tex_xyz);

		wet_sum = fmaf(fmaxf(density, 0.0), step_length / 10.0, wet_sum);

		ray_length += step_length;

	}

	dose->WETArray[vox_index] = wet_sum;

    __syncthreads();

}
