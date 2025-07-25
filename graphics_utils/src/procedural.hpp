#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "../../external/LiteMath/LiteMath.h"

namespace py = pybind11;

py::array_t<float> procedural_bsdf(py::array_t<float> uv, py::array_t<float> l_numpy, py::array_t<float> v_numpy, py::array_t<float> n_numpy);
