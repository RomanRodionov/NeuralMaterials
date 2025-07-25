#include "common.hpp"

float3 fresnelSchlick(float cosTheta, float3 F0) 
{
    return F0 + (float3(1.f, 1.f, 1.f) - F0) * std::pow(1.0f - cosTheta, 5.0f);
}

float ggxDistribution(float3 n, float3 h, float roughness) 
{
    float alpha = roughness * roughness;
    float alpha2 = alpha * alpha;
    float cosThetaH = std::max(LiteMath::dot(n, h), 0.0f);
    float denom = (cosThetaH * cosThetaH) * (alpha2 - 1.0f) + 1.0f;
    return alpha2 / (PI * denom * denom + EPSILON);
}

float schlickSmithG1(float3 n, float3 v, float roughness) 
{
    float k = (roughness + 1.0f) * (roughness + 1.0f) / 8.0f;
    float nv = std::max(LiteMath::dot(n, v), 0.0f);
    return nv / (nv * (1.0f - k) + k + EPSILON);
}

std::vector<float> real_fourier_moments_of(const std::vector<float> &phases, const std::vector<float> &values, int n)
{
    const unsigned N = phases.size();
    assert(N == values.size());
    int M = n - 1;
    std::vector<float> moments(M + 1);
    for(int i = 0; i <= M; ++i) {
        float val = 0.0f;
        for(unsigned j = 0; j < N; ++j) {
            val += values[j] * std::cos(float(i) * phases[j]);
        }
        moments[i] = val / float(N) * 2.f;
    }
    moments[0] = moments[0] * 0.5f;
    return moments;
}