#ifndef BASE_CLASSES_H
#define BASE_CLASSES_H

#include <stdio.h>
#include <stddef.h>
#include <cmath>
#include "./model_params.h"

#ifndef M_PI
#	define M_PI 3.14159265358979323846
#endif

#define LUT_LENGTH 400

typedef struct {

    size_t i;
    size_t j;
    size_t k;

} PointIJK;

typedef struct {

    float x;
    float y;
    float z;

} PointXYZ;

typedef struct {

    float x;
    float y;
    float mu;
    int energy_id;

} Spot;

typedef struct {

    int spot_start; // Starting spot index
    int n_spots;    // Total
    int energy_id;  // For indexing LUTs

    float r80;
    float energy;

} Layer;

typedef struct {

    float x1;
    float x2;
    float y_offset;
    float y_width;

} MLCPair;


class BeamClass {

    public:

        PointXYZ iso;
        PointXYZ src;
        float gantry_angle;
        float couch_angle;
        float collimator_angle;
        float mu;

        float singa, cosga; // Cached gantry angle trig functions
        float sinta, costa; // Cached couch angle trig functions
        float sinca, cosca; // Cached collimator angle trig functions

        int n_energies; // Total energies

        Layer * layers; // Layer information
        int n_layers;   // Number of non-empty layers

        Spot * spots;   // All spots, sorted by energy ID
        int n_spots;    // Spot count

        MLCPair * mlc;  // MLC leaf pairs
        int n_mlc_pairs;    // Number of leaf pairs

        float * divergence_params;  // R80, energy, coefficients
        int dvp_len;    // Length including R80 + energy (stride)

        float * lut_depths;
        float * lut_sigmas;
        float * lut_idds;
        int lut_len;

        // float * off_axis_radii;
        // float * off_axis_factors;
        // float * off_axis_softening;
        // int off_axis_len;

        BeamClass(float * iso, float gantry_angle, float couch_angle);

        BeamClass(float * iso, float gantry_angle, float couch_angle, float collimator_angle); 

        BeamClass(BeamClass * h_beam);

        void importLayers();

};


class DoseClass {

    public:

        PointIJK img_sz;
        unsigned int num_voxels;

        float spacing;

        float * DoseArray;
        float * DensityArray;
        float * WETArray;

        DoseClass(size_t img_sz[], float spacing);

        DoseClass(DoseClass * h_dose);

};


#endif