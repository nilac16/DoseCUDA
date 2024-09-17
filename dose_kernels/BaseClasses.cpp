#include "BaseClasses.h"
#include "MemoryClasses.h"


DoseClass::DoseClass(long int * img_sz, float spacing) {

	this->img_sz.i = img_sz[2];
	this->img_sz.j = img_sz[1];
	this->img_sz.k = img_sz[0];

	this->spacing = spacing;

	this->num_voxels = this->img_sz.i * this->img_sz.j * this->img_sz.k;

}

DoseClass::DoseClass(DoseClass * h_dose){
	this->img_sz = h_dose->img_sz;
	this->spacing = h_dose->spacing;
	this->num_voxels = h_dose->num_voxels;
}


BeamClass::BeamClass(float * iso, float gantry_angle, float couch_angle){

	this->iso.x = iso[0];
	this->iso.y = iso[1];
	this->iso.z = iso[2];
	this->gantry_angle = gantry_angle;
	this->couch_angle = couch_angle;

	//starting coordinate is [x, y, z] = [0.0, SAD, 0.0]
	//machine angles in radians
	float ga = gantry_angle * M_PI / 180.0f;
	float ta = couch_angle * M_PI / 180.0f;
	sincosf(ga, &this->singa, &this->cosga);
	sincosf(ta, &this->sinta, &this->costa);

	//gantry rotation - rotate about z-axis
	float xg, yg;
	xg = -((VSADX + VSADY) / 2.0) * this->singa;
	yg = ((VSADX + VSADY) / 2.0) * this->cosga;
	//zg = 0.0;

	//table rotation - rotate about y-axis
	float xt, yt, zt;
	xt = xg * this->costa;
	yt = yg;
	zt = -xg * this->sinta;

	//translate based on iso location in image
	this->src.x = xt;
	this->src.y = yt;
	this->src.z = zt;

}

BeamClass::BeamClass(BeamClass * h_beam){

	this->iso = h_beam->iso;
	this->src = h_beam->src;
	this->gantry_angle = h_beam->gantry_angle;
	this->couch_angle = h_beam->couch_angle;
	this->singa = h_beam->singa;
	this->cosga = h_beam->cosga;
	this->sinta = h_beam->sinta;
	this->costa = h_beam->costa;
	this->n_energies = h_beam->n_energies;
	this->n_layers = h_beam->n_layers;
	this->n_spots = h_beam->n_spots;
	this->dvp_len = h_beam->dvp_len;
	this->lut_len = h_beam->lut_len;

}


/** @brief Count spots in a subarray
 * 	@param spots
 * 		Spots array
 * 	@param n_spots
 * 		Size of the spots array
 * 	@param start
 * 		Beginning index
 * 	@param energy
 * 		Energy to count
 * 	@returns The number of spots starting from @p spots with energy ID @p energy
 */
static int count_spots(const Spot spots[], int n_spots, int start, int energy)
{
	int end;

	for (end = start; end < n_spots && spots[end].energy_id == energy; end++);
	return end - start;
}


void BeamClass::importLayers(){

	int spot_start = 0, spot_count;

	this->n_layers = 0;
	for (int energy_id = 0; energy_id < this->n_energies; energy_id++) {
		spot_count = count_spots(this->spots, this->n_spots, spot_start, energy_id);
		if (!spot_count) {
			/* No spots in this layer */
			continue;
		}

		Layer &layer = this->layers[this->n_layers];

		layer.spot_start = spot_start;
		layer.n_spots = spot_count;
		layer.energy_id = energy_id;

		layer.r80 = this->divergence_params[this->dvp_len * energy_id + 1];
		layer.energy = this->divergence_params[this->dvp_len * energy_id];

		this->n_layers++;

		spot_start += spot_count;
	}

}
