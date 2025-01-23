#include <algorithm>
#include "dosemodule.h"
#include "MemoryClasses.h"


/** @brief Check a NumPy array for dimensionality and contained element type
 * 	@param arr
 * 		NumPy array pointer
 * 	@param dim
 * 		Required dimensionality
 * 	@param type
 * 		Required type constant
 * 	@returns true if the constraint is satisfied, false if not
 */
static bool proton_array_typecheck(const PyArrayObject *arr,
								   int					dim,
								   int					type)
{
	return PyArray_NDIM(arr) == dim && PyArray_TYPE(arr) == type;
}


/** @brief Fetch the array's data pointer as a pointer to T */
template <class T>
static T *pyarray_as(PyArrayObject *arr)
{
	return reinterpret_cast<T *>(PyArray_DATA(arr));
}


extern void proton_raytrace_cuda(int gpu_id, DoseClass * h_dose, BeamClass * h_beam);


extern void proton_spot_cuda(int gpu_id, DoseClass * h_dose, BeamClass * h_beam);


extern void photon_dose_cuda(int gpu_id, DoseClass * h_dose, BeamClass * h_beam);


static PyObject* proton_raytrace(PyObject *self, PyObject *args) {

	PyArrayObject *density_array, *iso;
	double gantry_angle, couch_angle, spacing;
	int gpu_id;

	if (!PyArg_ParseTuple(args, "O!O!dddi",
			&PyArray_Type, &density_array,
			&PyArray_Type, &iso,
			&gantry_angle,
			&couch_angle,
			&spacing,
			&gpu_id))
		return NULL ;

	if (!proton_array_typecheck(density_array, 3, NPY_FLOAT)) {
		PyErr_SetString(PyExc_ValueError, "Density array must be three-dimensional and of type float.");
		return NULL ;
	}

	if (!proton_array_typecheck(iso, 1, NPY_FLOAT)) {
		PyErr_SetString(PyExc_ValueError, "Isocenter array must be one-dimensional and of type float.");
		return NULL ;
	}

	float adjusted_gantry_angle = fmodf(gantry_angle + 180.0f, 360.0f);

	try {

		size_t dims[3] = {
			(size_t)PyArray_DIMS(density_array)[0],
			(size_t)PyArray_DIMS(density_array)[1],
			(size_t)PyArray_DIMS(density_array)[2],
		};

		DoseClass dose_obj = DoseClass(dims, spacing);
		HostPointer<float> WETArray(dose_obj.num_voxels);

		dose_obj.DensityArray = pyarray_as<float>(density_array);
		dose_obj.WETArray = WETArray.get();

		BeamClass beam_obj = BeamClass(pyarray_as<float>(iso), adjusted_gantry_angle, couch_angle);

		proton_raytrace_cuda(gpu_id, &dose_obj, &beam_obj);

		PyObject *return_wet = PyArray_SimpleNewFromData(3, PyArray_DIMS(density_array), PyArray_TYPE(density_array), WETArray.release());

		PyArray_ENABLEFLAGS((PyArrayObject*) return_wet, NPY_ARRAY_OWNDATA);

		return return_wet;

	} catch (std::bad_alloc &) {

		PyErr_SetString(PyExc_MemoryError, "Not enough host memory");

	} catch (std::runtime_error &e) {

		PyErr_Format(PyExc_RuntimeError, "CUDA error: %s", e.what());

	}

	return NULL;

}


static bool spot_compare(const Spot &a, const Spot &b)
{
	return a.energy_id < b.energy_id;
}


/** Create the authoritative spot array from spot data in any order */
static void make_spot_array(PyArrayObject *spots, HostPointer<Spot> &res)
{
	const size_t count = PyArray_DIM(spots, 0);
	const float *src = pyarray_as<float>(spots);

	for (size_t i = 0; i < count; i++) {
		res[i].x = src[4 * i];
		res[i].y = src[4 * i + 1];
		res[i].mu = src[4 * i + 2];
		res[i].energy_id = static_cast<int>(src[4 * i + 3]);
	}

	std::sort(&res[0], &res[count], spot_compare);
}


static void make_mlc_array(PyArrayObject *mlc, HostPointer<MLCPair> &res)
{
	const size_t count = PyArray_DIM(mlc, 0);
	const float *src = pyarray_as<float>(mlc);

	for (size_t i = 0; i < count; i++) {
		res[i].x1 = src[i];
		res[i].x2 = src[i + count];
		res[i].y_offset = src[i + 2 * count];
		res[i].y_width = src[i + 3 * count];
		// printf("MLC Pair %d: x1: %f, x2: %f, y_offset: %f, y_width: %f\n", i, res[i].x1, res[i].x2, res[i].y_offset, res[i].y_width);
	}
}


static PyObject* proton_spot(PyObject *self, PyObject *args) {

	PyArrayObject *wet_array, *density_array, *isocenter, *spots, *lut_depths, *lut_sigmas, *lut_idds, *lut_divergence_params;
	double gantry_angle, couch_angle, density_spacing;
	int gpu_id;

	if (!PyArg_ParseTuple(args, "O!O!O!ddO!dO!O!O!O!i",
			&PyArray_Type, &wet_array,
			&PyArray_Type, &density_array,
			&PyArray_Type, &isocenter,
			&gantry_angle,
			&couch_angle,
			&PyArray_Type, &spots,
			&density_spacing,
			&PyArray_Type, &lut_depths,
			&PyArray_Type, &lut_sigmas,
			&PyArray_Type, &lut_idds,
			&PyArray_Type, &lut_divergence_params,
			&gpu_id))
		return NULL ;

	if (!proton_array_typecheck(wet_array, 3, NPY_FLOAT)) {
		PyErr_SetString(PyExc_ValueError, "Density array must be three-dimensional and of type float.");
		return NULL ;
	}

	if (!proton_array_typecheck(isocenter, 1, NPY_FLOAT)) {
		PyErr_SetString(PyExc_ValueError, "Isocenter array must be one-dimensional and of type float.");
		return NULL ;
	}

	if (!proton_array_typecheck(spots, 2, NPY_FLOAT)) {
		PyErr_SetString(PyExc_ValueError, "Spot data array must be two-dimensional and of type float.");
		return NULL ;
	}

	float adjusted_gantry_angle = fmodf(gantry_angle + 180.0f, 360.0f);
	size_t n_energies = PyArray_DIM(lut_depths, 0);
	size_t n_spots = PyArray_DIM(spots, 0);

	try {

		// Create dose
		size_t dims[3] = {
			(size_t)PyArray_DIMS(wet_array)[0],
			(size_t)PyArray_DIMS(wet_array)[1],
			(size_t)PyArray_DIMS(wet_array)[2],
		};

		DoseClass dose_obj = DoseClass(dims, density_spacing);
		HostPointer<float> DoseArray(dose_obj.num_voxels);

		dose_obj.DoseArray = DoseArray.get();
		dose_obj.DensityArray = pyarray_as<float>(density_array);
		dose_obj.WETArray = pyarray_as<float>(wet_array);

		// Create beam
		BeamClass beam_obj = BeamClass(pyarray_as<float>(isocenter), adjusted_gantry_angle, couch_angle);
		HostPointer<Layer> LayerArray(n_energies);
		HostPointer<Spot> SpotArray(n_spots);

		make_spot_array(spots, SpotArray);

		beam_obj.n_energies = n_energies;
		beam_obj.layers = LayerArray.get();
		beam_obj.spots = SpotArray.get();
		beam_obj.n_spots = n_spots;
		beam_obj.divergence_params = pyarray_as<float>(lut_divergence_params);
		beam_obj.dvp_len = 5;	// R80, energy, quadratic coefficients
		beam_obj.lut_depths = pyarray_as<float>(lut_depths);
		beam_obj.lut_sigmas = pyarray_as<float>(lut_sigmas);
		beam_obj.lut_idds = pyarray_as<float>(lut_idds);
		beam_obj.lut_len = LUT_LENGTH;	// This can now be changed at runtime

		beam_obj.importLayers();

		proton_spot_cuda(gpu_id, &dose_obj, &beam_obj);

		PyObject *return_dose = PyArray_SimpleNewFromData(3, PyArray_DIMS(wet_array), PyArray_TYPE(wet_array), DoseArray.release());

		PyArray_ENABLEFLAGS((PyArrayObject*) return_dose, NPY_ARRAY_OWNDATA);

		return return_dose;

	} catch (std::bad_alloc &) {

		PyErr_SetString(PyExc_MemoryError, "Not enough host memory");

	} catch (std::runtime_error &e) {

		PyErr_Format(PyExc_RuntimeError, "CUDA error: %s", e.what());

	}

	return NULL;

}


static PyObject * photon_dose(PyObject* self, PyObject* args) {

	PyArrayObject *density_array, *mlc, *isocenter;
    double mu, gantry_angle, collimator_angle, couch_angle, jaw1, jaw2, voxel_sp;
    int gpu_id;

    if (!PyArg_ParseTuple(args, "O!O!dO!ddddddi", 
			&PyArray_Type, &density_array, 
			&PyArray_Type, &isocenter, 
			&mu, 
			&PyArray_Type, &mlc,
    		&gantry_angle, 
			&collimator_angle, 
			&couch_angle, 
			&jaw1, 
			&jaw2, 
			&voxel_sp, 
			&gpu_id))
        return NULL;

	if (!proton_array_typecheck(density_array, 3, NPY_FLOAT)) {
		PyErr_SetString(PyExc_ValueError, "Density array must be three-dimensional and of type float.");
		return NULL ;
	}

	if (!proton_array_typecheck(isocenter, 1, NPY_FLOAT)) {
		PyErr_SetString(PyExc_ValueError, "Isocenter array must be one-dimensional and of type float.");
		return NULL ;
	}

	if (!proton_array_typecheck(mlc, 2, NPY_FLOAT)) {
		PyErr_SetString(PyExc_ValueError, "MLC array must be two-dimensional and of type float.");
		return NULL ;
	}

	float adjusted_gantry_angle = fmodf(gantry_angle + 180.0f, 360.0f);
	size_t n_mlc_pairs = PyArray_DIM(mlc, 0);

	try {

		size_t dims[3] = {
			(size_t)PyArray_DIMS(density_array)[0],
			(size_t)PyArray_DIMS(density_array)[1],
			(size_t)PyArray_DIMS(density_array)[2],
		};

		DoseClass dose_obj = DoseClass(dims, voxel_sp);
		HostPointer<float> WETArray(dose_obj.num_voxels);
		HostPointer<float> DoseArray(dose_obj.num_voxels);

		dose_obj.DensityArray = pyarray_as<float>(density_array);
		dose_obj.WETArray = WETArray.get();
		dose_obj.DoseArray = DoseArray.get();

		BeamClass beam_obj = BeamClass(pyarray_as<float>(isocenter), adjusted_gantry_angle, couch_angle, collimator_angle);
		HostPointer<MLCPair> MLCPairArray(n_mlc_pairs);
		make_mlc_array(mlc, MLCPairArray);
		beam_obj.n_mlc_pairs = n_mlc_pairs;
		beam_obj.mu = mu;
		beam_obj.mlc = MLCPairArray.get();

    	photon_dose_cuda(gpu_id, &dose_obj, &beam_obj);

		PyObject *return_dose = PyArray_SimpleNewFromData(3, PyArray_DIMS(density_array), PyArray_TYPE(density_array), DoseArray.release());

		PyArray_ENABLEFLAGS((PyArrayObject*) return_dose, NPY_ARRAY_OWNDATA);

		return return_dose;

	} catch (std::bad_alloc &) {

		PyErr_SetString(PyExc_MemoryError, "Not enough host memory");

	} catch (std::runtime_error &e) {

		PyErr_Format(PyExc_RuntimeError, "CUDA error: %s", e.what());

	}

	return NULL;

}


static PyMethodDef DoseMethods[] = {
	{
		"proton_raytrace_cuda",
		proton_raytrace,
		METH_VARARGS,
		"Compute WET array for proton dose calc."
	},
	{
		"proton_spot_cuda",
		proton_spot,
		METH_VARARGS,
		"Compute proton spot dose with PB using pre-calc'd WET array."
	},
	{
		"photon_dose_cuda",
		photon_dose,
		METH_VARARGS,
		"Compute proton spot dose with PB using pre-calc'd WET array."
	},
	{ 0 }
};


static struct PyModuleDef dosemodule = {
	PyModuleDef_HEAD_INIT,
	"dose_kernels",
	"Compute dose on the GPU.",
	-1,
	DoseMethods,
};


PyMODINIT_FUNC PyInit_dose_kernels(void) {
	import_array();
	return PyModule_Create(&dosemodule);
}
