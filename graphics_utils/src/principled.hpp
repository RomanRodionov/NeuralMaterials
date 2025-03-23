#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "../../external/LiteMath/LiteMath.h"

namespace py = pybind11;

py::array_t<float> principled_bsdf(py::array_t<float> i_numpy, py::array_t<float> o_numpy, py::array_t<float> n_numpy, 
                      py::array_t<float> base_color, float metallic, float roughness, py::array_t<float> specular);