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


__host__ CudaDose::CudaDose(DoseClass * h_dose) : DoseClass(h_dose) {}

__device__ bool CudaDose::pointIJKWithinImage(const PointIJK * point_ijk) {

	return point_ijk->i < this->img_sz.i
	    && point_ijk->j < this->img_sz.j
		&& point_ijk->k < this->img_sz.k;
}

__device__ unsigned int CudaDose::pointIJKtoIndex(const PointIJK * point_ijk) {

	return point_ijk->i + this->img_sz.i * (point_ijk->j + this->img_sz.j * point_ijk->k);
}

__device__ void CudaDose::pointIJKtoXYZ(const PointIJK * point_ijk, PointXYZ * point_xyz, BeamClass * beam) {

	point_xyz->x = (float)point_ijk->i * this->spacing - beam->iso.x;
	point_xyz->y = (float)point_ijk->j * this->spacing - beam->iso.y;
	point_xyz->z = (float)point_ijk->k * this->spacing - beam->iso.z;

}

__device__ void CudaDose::pointXYZtoIJK(const PointXYZ * point_xyz, PointIJK * point_ijk, BeamClass * beam) {

	point_ijk->i = (int)roundf((point_xyz->x + beam->iso.x) / this->spacing);
	point_ijk->j = (int)roundf((point_xyz->y + beam->iso.y) / this->spacing);
	point_ijk->k = (int)roundf((point_xyz->z + beam->iso.z) / this->spacing);
}

/** sincos but with a value theoretically supplied to arctangent */
__device__ void sincos_from_atan(float y, float x, float *sptr, float *cptr) {

	float slope = y / x;

	*cptr = rsqrtf(fmaf(slope, slope, 1.0));
	*sptr = slope * *cptr;
}

__device__ void CudaDose::pointXYZImageToHead(const PointXYZ * point_img, PointXYZ * point_head, BeamClass * beam){

	float sinx, cosx;

	//table rotation - rotate about y-axis (negative direction)
	float xt, yt, zt;
	sinx = beam->sinta;
	cosx = beam->costa;
	xt = point_img->x * cosx + point_img->z * sinx;
	yt = point_img->y;
	zt = -point_img->x * sinx + point_img->z * cosx;

	//gantry rotation - rotate about z-axis (negative direction)
	float xg, yg, zg;
	sinx = -beam->singa;
	cosx = beam->cosga;
	xg = xt * cosx - yt * sinx;
	yg = xt * sinx + yt * cosx;
	zg = zt;

	// //spot steering in z - rotate about x
	// float xz, yz, zz;
	// /* float z_rot = atan2f(spot->y, VSADY);
	// sincosf(z_rot, &sinx, &cosx); */

	// sincos_from_atan(spot->y, VSADY, &sinx, &cosx);
	// yg = yg - VSADY;
	// xz = xg;
	// yz = yg * cosx - zg * sinx;
	// zz = yg * sinx + zg * cosx;
	// yz = yz + VSADY;

	// //spot steering in x - rotate about z
	// float xx, yx, zx;
	// /* float x_rot = atan2f(spot->x, VSADX);
	// sincosf(x_rot, &sinx, &cosx); */

	// sincos_from_atan(spot->x, VSADX, &sinx, &cosx);
	// yz = yz - VSADX;
	// xx = xz * cosx - yz * sinx;
	// yx = xz * sinx + yz * cosx;
	// zx = zz;
	// yx = yx + VSADX;

	point_head->x = xg;
	point_head->y = yg;
	point_head->z = zg;

}

__device__ void CudaDose::pointXYZHeadToImage(const PointXYZ * point_head, PointXYZ * point_img, BeamClass * beam){

	float sinx, cosx;

	// //spot steering in x - rotate about z
	// float xx, yx, zx, y_;
	// /* float x_rot = -atan2f(spot->x , VSADX);
	// sincosf(x_rot, &sinx, &cosx); */
	// sincos_from_atan(-spot->x, VSADX, &sinx, &cosx);
	// y_ = point_head->y - VSADX;

	// xx = point_head->x * cosx - y_ * sinx;
	// yx = point_head->x * sinx + y_ * cosx;
	// zx = point_head->z;

	// yx = yx + VSADX - VSADY;

	// //spot steering in z - rotate about x
	// float xz, zz, yz;
	// /* float z_rot = -atan2f(spot->y , VSADY);
	// sincosf(z_rot, &sinx, &cosx); */
	// sincos_from_atan(-spot->y, VSADY, &sinx, &cosx);
	// xz = xx;
	// yz = yx * cosx - zx * sinx;
	// zz = yx * sinx + zx * cosx;

	// yz = yz + VSADY;

	float xz = point_head->x;
	float yz = point_head->y;
	float zz = point_head->z;

	//gantry rotation - rotate about z-axis (negative direction)
	float xg, yg, zg;
	sinx = beam->singa;
	cosx = beam->cosga;
	xg = xz * cosx - yz * sinx;
	yg = xz * sinx + yz * cosx;
	zg = zz;

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