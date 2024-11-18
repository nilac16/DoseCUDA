#ifndef DOSEMODULE_H
#define DOSEMODULE_H

#include <stdio.h>
#include <cmath>

#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <object.h>
#include <numpy/arrayobject.h>
#include "BaseClasses.h"

extern void proton_raytrace_cuda(int gpu_id, DoseClass * h_dose, BeamClass * h_beam);

extern void proton_spot_cuda(int gpu_id, DoseClass * h_dose, BeamClass * h_beam);

#endif
