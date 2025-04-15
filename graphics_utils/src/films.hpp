#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "../../external/LiteMath/LiteMath.h"

namespace py = pybind11;

float film_refl(py::array_t<float> w_i_numpy, py::array_t<float> w_o_numpy, py::array_t<float> etaI_numpy, py::array_t<float> etaF_numpy, py::array_t<float>  etaT_numpy, float thickness, float lambda);