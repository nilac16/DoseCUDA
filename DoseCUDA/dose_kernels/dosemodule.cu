#include <algorithm>

#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <object.h>
#include <numpy/arrayobject.h>

#include "IMRTClasses.cuh"
#include "IMPTClasses.cuh"
#include "MemoryClasses.h"
#include "model_params.h"


/** @brief Check a NumPy array for dimensionality and contained element type
 * 	@param arr
 * 		NumPy array pointer
 * 	@param dim
 * 		Required dimensionality
 * 	@param type
 * 		Required type constant
 * 	@returns true if the constraint is satisfied, false if not
 */
static bool proton_array_typecheck(const PyArrayObject *arr, int dim, int type) {

	return PyArray_NDIM(arr) == dim && PyArray_TYPE(arr) == type;
}


/** @brief Fetch the array's data pointer as a pointer to T */
template <class T>
static T *pyarray_as(PyArrayObject *arr) {

	return reinterpret_cast<T *>(PyArray_DATA(arr));
}


static IMPTBeam::Model proton_beam_model(void/* PyObject * model */) {

	IMPTBeam::Model model;

	model.vsadx = VSADX;
	model.vsady = VSADY;
	return model;
}


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

		CudaDose dose_obj = CudaDose(dims, spacing);
		HostPointer<float> WETArray(dose_obj.num_voxels);

		dose_obj.DensityArray = pyarray_as<float>(density_array);
		dose_obj.WETArray = WETArray.get();

		auto model = proton_beam_model();
		IMPTBeam beam_obj = IMPTBeam(pyarray_as<float>(iso), adjusted_gantry_angle, couch_angle, &model);

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

		IMPTDose dose_obj = IMPTDose(dims, density_spacing);
		HostPointer<float> DoseArray(dose_obj.num_voxels);

		dose_obj.DoseArray = DoseArray.get();
		dose_obj.DensityArray = pyarray_as<float>(density_array);
		dose_obj.WETArray = pyarray_as<float>(wet_array);

		// Create beam
		auto model = proton_beam_model();
		IMPTBeam beam_obj = IMPTBeam(pyarray_as<float>(isocenter), adjusted_gantry_angle, couch_angle, &model);
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


static IMRTBeam::Model photon_beam_model(void/* PyObject * model */) {

	IMRTBeam::Model model;

	model.air_density				= AIR_DENSITY;
	model.mu_cal					= MU_CAL;
	model.primary_src_dist			= PRIMARY_SOURCE_DISTANCE;
	model.scatter_src_dist			= SCATTER_SOURCE_DISTANCE;
	model.mlc_distance				= MLC_DISTANCE;
	model.scatter_src_weight		= SCATTER_SOURCE_WEIGHT;
	model.electron_mass_attenuation	= ELECTRON_MASS_ATTENUATION;
	return model;
}


static PyObject * photon_dose(PyObject* self, PyObject* args) {

	PyObject *model_instance, *volume_instance, *cp_instance;
	int gpu_id;	

	// parse arguments
    if (!PyArg_ParseTuple(args, "OOOi", &model_instance, &volume_instance, &cp_instance, &gpu_id))
        return NULL;

	PyObject *model_class = PyObject_GetAttrString(PyImport_ImportModule("DoseCUDA.plan_imrt"), "IMRTBeamModel");
    if (!model_class || !PyObject_IsInstance(model_instance, model_class)) {
        PyErr_SetString(PyExc_TypeError, "Argument 1 must be an instance of IMRTBeamModel");
        return NULL;
    }

	PyObject *volume_class = PyObject_GetAttrString(PyImport_ImportModule("DoseCUDA.plan"), "VolumeObject");
    if (!volume_class || !PyObject_IsInstance(volume_instance, volume_class)) {
        PyErr_SetString(PyExc_TypeError, "Argument 2 must be an instance of VolumeObject");
        return NULL;
    }

	PyObject *cp_class = PyObject_GetAttrString(PyImport_ImportModule("DoseCUDA.plan_imrt"), "IMRTControlPoint");
    if (!cp_class || !PyObject_IsInstance(cp_instance, cp_class)) {
        PyErr_SetString(PyExc_TypeError, "Argument 3 must be an instance of IMRTControlPoint");
        return NULL;
    }

	// check volume data properties
	PyObject *density_object = PyObject_GetAttrString(volume_instance, "voxel_data");
    if (!density_object) {
        PyErr_SetString(PyExc_AttributeError, "VolumeObject instance has no attribute 'voxel_data'");
        return NULL;
    }

	PyArrayObject *density_array = (PyArrayObject *) density_object;
	if (!proton_array_typecheck(density_array, 3, NPY_FLOAT)) {
		PyErr_SetString(PyExc_ValueError, "'voxel_data' must be three-dimensional and of type float.");
		return NULL ;
	}

	PyObject *spacing_object = PyObject_GetAttrString(volume_instance, "spacing");
    if (!spacing_object) {
        PyErr_SetString(PyExc_AttributeError, "VolumeObject instance has no attribute 'spacing'");
        return NULL;
    }
	
	PyArrayObject *spacing_array = (PyArrayObject *) spacing_object;
	if (!proton_array_typecheck(spacing_array, 1, NPY_FLOAT)) {
		PyErr_SetString(PyExc_ValueError, "'spacing' must be one-dimensional and of type float.");
		return NULL ;
	}

	PyObject *origin_object = PyObject_GetAttrString(volume_instance, "origin");
    if (!origin_object) {
        PyErr_SetString(PyExc_AttributeError, "VolumeObject instance has no attribute 'origin'");
        return NULL;
    }

	PyArrayObject *origin_array = (PyArrayObject *) origin_object;
	if (!proton_array_typecheck(origin_array, 1, NPY_FLOAT)) {
		PyErr_SetString(PyExc_ValueError, "'origin' must be one-dimensional and of type float.");
		return NULL ;
	}

	// check control point data properties
	PyObject *iso_object = PyObject_GetAttrString(cp_instance, "iso");
    if (!iso_object) {
        PyErr_SetString(PyExc_AttributeError, "IMRTControlPoint instance has no attribute 'iso'");
        return NULL;
    }

	PyArrayObject *iso_array = (PyArrayObject *) iso_object;
	if (!proton_array_typecheck(iso_array, 1, NPY_FLOAT)) {
		PyErr_SetString(PyExc_ValueError, "'iso' must be one-dimensional and of type float.");
		return NULL ;
	}

	PyObject *mlc_object = PyObject_GetAttrString(cp_instance, "mlc");
    if (!mlc_object) {
        PyErr_SetString(PyExc_AttributeError, "IMRTControlPoint instance has no attribute 'mlc'");
        return NULL;
    }

	PyArrayObject *mlc_array = (PyArrayObject *) mlc_object;
	if (!proton_array_typecheck(mlc_array, 2, NPY_FLOAT)) {
		PyErr_SetString(PyExc_ValueError, "'mlc' must be two-dimensional and of type float.");
		return NULL ;
	}

	PyObject *mu_object = PyObject_GetAttrString(cp_instance, "mu");
    if (!mu_object) {
        PyErr_SetString(PyExc_AttributeError, "IMRTControlPoint instance has no attribute 'mu'");
        return NULL;
    }

	double mu = PyFloat_AsDouble(mu_object);
	Py_DECREF(mu_object);

	PyObject *gantry_object = PyObject_GetAttrString(cp_instance, "ga");
	if (!gantry_object) {
		PyErr_SetString(PyExc_AttributeError, "IMRTControlPoint instance has no attribute 'ga'");
		return NULL;
	}

	double ga = PyFloat_AsDouble(gantry_object);
	Py_DECREF(gantry_object);

	PyObject *collimator_object = PyObject_GetAttrString(cp_instance, "ca");
	if (!collimator_object) {
		PyErr_SetString(PyExc_AttributeError, "IMRTControlPoint instance has no attribute 'ca'");
		return NULL;
	}

	double ca = PyFloat_AsDouble(collimator_object);
	Py_DECREF(collimator_object);

	PyObject *couch_object = PyObject_GetAttrString(cp_instance, "ta");
	if (!couch_object) {
		PyErr_SetString(PyExc_AttributeError, "IMRTControlPoint instance has no attribute 'ta'");
		return NULL;
	}

	double ta = PyFloat_AsDouble(couch_object);
	Py_DECREF(couch_object);

	// beam model 
	PyObject *mu_cal_object = PyObject_GetAttrString(model_instance, "mu_calibration");
	if (!mu_cal_object) {
		PyErr_SetString(PyExc_AttributeError, "IMRTBeamModel instance has no attribute 'mu_calibration'");
		return NULL;
	}

	double mu_cal = PyFloat_AsDouble(mu_cal_object);
	Py_DECREF(mu_cal_object);

	float * spacing = pyarray_as<float>(spacing_array);
	float * origin = pyarray_as<float>(origin_array);
	float * iso = pyarray_as<float>(iso_array);
	double voxel_sp = (double)spacing[0];

	size_t n_mlc_pairs = PyArray_DIM(mlc_array, 0);

	try {

		float adjusted_ga = fmodf(ga + 180.0f, 360.0f);

		size_t dims[3] = {
			(size_t)PyArray_DIMS(density_array)[0],
			(size_t)PyArray_DIMS(density_array)[1],
			(size_t)PyArray_DIMS(density_array)[2],
		};

		float adjusted_isocenter[3] = {
			iso[0] - origin[0],
			iso[1] - origin[1],
			iso[2] - origin[2]
		};

		IMRTDose dose_obj = IMRTDose(dims, voxel_sp);
		HostPointer<float> WETArray(dose_obj.num_voxels);
		HostPointer<float> DoseArray(dose_obj.num_voxels);

		dose_obj.DensityArray = pyarray_as<float>(density_array);
		dose_obj.WETArray = WETArray.get();
		dose_obj.DoseArray = DoseArray.get();

		auto model = photon_beam_model();
		IMRTBeam beam_obj = IMRTBeam(adjusted_isocenter, adjusted_ga, ta, ca, &model);
		HostPointer<MLCPair> MLCPairArray(n_mlc_pairs);
		make_mlc_array(mlc_array, MLCPairArray);
		beam_obj.n_mlc_pairs = n_mlc_pairs;
		beam_obj.mlc = MLCPairArray.get();
		beam_obj.mu = mu;

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
