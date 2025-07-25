#include "procedural.hpp"
#include "common.hpp"

float3 proceduralWood(float2 uv, float3 N, float3 L, float3 V) {
    float r = length(uv - 0.5);
    float rings = sin(r * 12.0 + noise(uv * 5.0) * 2.0);

    float3 base = mix(float3(0.6, 0.4, 0.2), float3(0.3, 0.2, 0.1), rings);

    float rough = mix(0.6, 0.2, rings);

    float3 H = normalize(L + V);
    float spec = pow(max(dot(N, H), 0.0), 16.0);

    return base * max(dot(N, L), 0.0) + float3(spec) * 0.1 * rough;
}

py::array_t<float> procedural_bsdf(py::array_t<float> uv_numpy, py::array_t<float> l_numpy, py::array_t<float> v_numpy, py::array_t<float> n_numpy)
{
    py::buffer_info uv_info = uv_numpy.request();
    float2* uv = static_cast<float2*>(l_info.ptr);

    py::buffer_info l_info = l_numpy.request();
    float3* l = static_cast<float3*>(l_info.ptr);

    py::buffer_info V_info = V_numpy.request();
    float3* V = static_cast<float3*>(V_info.ptr);

    py::buffer_info N_info = N_numpy.request();
    float3* N = static_cast<float3*>(N_info.ptr);

    float3 bsdfValue = proceduralWood(*uv, *N, *l, *V);

    py::array_t<float> result(3);
    auto result_buf = result.request();
    float* result_ptr = static_cast<float*>(result_buf.ptr);

    result_ptr[0] = bsdfValue[0];
    result_ptr[1] = bsdfValue[1];
    result_ptr[2] = bsdfValue[2];

    return result;
}