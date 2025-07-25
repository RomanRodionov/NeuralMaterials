#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "films.hpp"
#include "principled.hpp"
#include "common.hpp"
//#include "procedural.hpp"

namespace py = pybind11;

py::array_t<float> real_fourier_moments(py::array_t<float> phases, py::array_t<float> values, uint32_t n)
{
    py::buffer_info phases_info = phases.request();
    float *phases_ptr = static_cast<float *>(phases_info.ptr);

    py::buffer_info values_info = values.request();
    float *values_ptr = static_cast<float *>(values_info.ptr);

    if(phases_info.ndim != 1 || values_info.ndim != 1)
        throw std::runtime_error("Number of dimensions must be one");
    if(phases_info.size != values_info.size) 
        throw std::runtime_error("phases and values have different sizes");

    std::vector<float> res = real_fourier_moments_of(std::vector<float>(phases_ptr, phases_ptr + phases_info.size),
                                                                 std::vector<float>(values_ptr, values_ptr + phases_info.size), n);

    py::array_t<float> result(res.size());
    auto result_info = result.request();
    float* result_ptr = static_cast<float *>(result_info.ptr);
    std::copy(res.begin(), res.end(), result_ptr);
    return result;
}

PYBIND11_MODULE(graphics_utils, m) {
    m.doc() = "Backend for intensive graphics routine";
    m.def("real_fourier_moments",    &real_fourier_moments,    "Calculate fourier moments of spectrum");
    m.def("film_refl",       &film_refl,       "Calculate reflection intensity for thin film");
    m.def("principled_bsdf", &principled_bsdf, "Calculate BSDF for principled material"      );
    //m.def("procedural_bsdf", &procedural_bsdf, "Calculate BSDF for procedural material"      );
}