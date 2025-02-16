#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "films.hpp"

namespace py = pybind11;

PYBIND11_MODULE(graphics_utils, m) {
    m.doc() = "Backend for highly demanding parts";
    m.def("film_refl", &film_refl, "Calculate reflection intensity for thin film");
}