#include "principled.hpp"
#include "common.hpp"

float3 cookTorranceBRDF(float3 base_color, float metallic, float roughness, float3 specular,
                        float3  normal, float3 viewDir, float3 lightDir) 
{
    float3 halfVector = LiteMath::normalize(viewDir + lightDir);

    float3 F0_dielectric = specular;
    float3 F0_metal = base_color;
    float3 F0 = F0_dielectric * (1.0f - metallic) + F0_metal * metallic;

    float cosThetaH = std::max(LiteMath::dot(halfVector, viewDir), 0.0f);
    float3 fresnel = fresnelSchlick(cosThetaH, F0);

    float ndf = ggxDistribution(normal, halfVector, roughness);
    float g = schlickSmithG1(normal, viewDir, roughness) * schlickSmithG1(normal, lightDir, roughness);

    float3 specularReflection = (fresnel * ndf * g) / (4.0f * std::max(LiteMath::dot(normal, viewDir), 0.0f) * std::max(LiteMath::dot(normal, lightDir), 0.0f) + EPSILON);
    float3 diffuse = ((float3(1.f, 1.f, 1.f) - fresnel) * (1.0f - metallic) * base_color) / PI;

    return diffuse + specularReflection;
}

py::array_t<float> principled_bsdf(py::array_t<float> w_i_numpy, py::array_t<float> w_o_numpy, py::array_t<float> w_n_numpy, 
    py::array_t<float> base_color_numpy, float metallic, float roughness, py::array_t<float> specular_numpy)
{
    py::buffer_info w_i_info = w_i_numpy.request();
    float3* w_i = static_cast<float3*>(w_i_info.ptr);

    py::buffer_info w_o_info = w_o_numpy.request();
    float3* w_o = static_cast<float3*>(w_o_info.ptr);

    py::buffer_info w_n_info = w_n_numpy.request();
    float3* w_n = static_cast<float3*>(w_n_info.ptr);

    py::buffer_info base_color_info = base_color_numpy.request();
    float3* base_color = static_cast<float3*>(base_color_info.ptr);

    py::buffer_info specular_info = specular_numpy.request();
    float3* specular = static_cast<float3*>(specular_info.ptr);

    float3 bsdfValue = cookTorranceBRDF(*base_color, metallic, roughness, *specular, *w_n, *w_i, *w_o);

    py::array_t<float> result(3);
    auto result_buf = result.request();
    float* result_ptr = static_cast<float*>(result_buf.ptr);

    result_ptr[0] = bsdfValue[0];
    result_ptr[1] = bsdfValue[1];
    result_ptr[2] = bsdfValue[2];

    return result;
}