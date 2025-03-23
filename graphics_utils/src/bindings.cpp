#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "films.hpp"
#include "principled.hpp"

namespace py = pybind11;

PYBIND11_MODULE(graphics_utils, m) {
    m.doc() = "Backend for intensive graphics routine";
    m.def("film_refl",       &film_refl,       "Calculate reflection intensity for thin film");
    m.def("principled_bsdf", &principled_bsdf, "Calculate BSDF for principled material"      );
}