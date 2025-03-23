#include "../../external/LiteMath/LiteMath.h"

using LiteMath::complex, LiteMath::M_PI, LiteMath::float3;

constexpr float PI = 3.14159265359f;
constexpr float EPSILON = 1e-6f;

float3 fresnelSchlick(float cosTheta, float3 F0);

float ggxDistribution(float3 n, float3 h, float roughness);

float schlickSmithG1(float3 n, float3 v, float roughness);