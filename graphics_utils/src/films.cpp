#include "films.hpp"

using LiteMath::complex, LiteMath::M_PI, LiteMath::float3;

//thin film reflections are baised on Airy formulas

inline complex FrComplexReflS(complex cosThetaI, complex cosThetaT, complex iorI, complex iorT)
{
  if (complex_norm(cosThetaI) < 1e-6f) //approximated
  {
    return {-1, 0};
  }
  return (iorI * cosThetaI - iorT * cosThetaT) / (iorI * cosThetaI + iorT * cosThetaT);
}

inline complex FrComplexReflP(complex cosThetaI, complex cosThetaT, complex iorI, complex iorT)
{
  if (complex_norm(cosThetaI) < 1e-6f) //approximated
  {
    return {-1, 0};
  }
  return (iorT * cosThetaI - iorI * cosThetaT) / (iorT * cosThetaI + iorI * cosThetaT);
}

inline complex calcPhaseDiff(complex cosTheta, complex eta, float thickness, float lambda)
{
  return 4 * M_PI * eta * cosTheta * thickness / complex(lambda);
}

float FrFilmRefl(float cosThetaI, complex etaI, complex etaF, complex etaT, float thickness, float lambda)
{
  complex sinThetaI = 1.0f - cosThetaI * cosThetaI;
  complex sinThetaF = sinThetaI * (etaI.re * etaI.re) / (etaF * etaF);
  complex cosThetaF = complex_sqrt(1.0f - sinThetaF);
  complex sinThetaT = sinThetaI * (etaI.re * etaI.re) / (etaT * etaT);
  complex cosThetaT = complex_sqrt(1.0f - sinThetaT);
  
  complex phaseDiff = calcPhaseDiff(cosThetaF, etaF, thickness, lambda);

  float result = 0;
    
  complex FrReflI = FrComplexReflS(cosThetaI, cosThetaF, etaI, etaF);
  complex FrReflF = FrComplexReflS(cosThetaF, cosThetaT, etaF, etaT);
  complex FrRefl  = FrReflF * std::exp(-phaseDiff.im) * complex(std::cos(phaseDiff.re), std::sin(phaseDiff.re));
  FrRefl          = (FrReflI + FrRefl) / (1 + FrReflI * FrRefl);
  result += complex_norm(FrRefl);

  FrReflI = FrComplexReflP(cosThetaI, cosThetaF, etaI, etaF);
  FrReflF = FrComplexReflP(cosThetaF, cosThetaT, etaF, etaT);
  FrRefl  = FrReflF * std::exp(-phaseDiff.im) * complex(std::cos(phaseDiff.re), std::sin(phaseDiff.re));
  FrRefl  = (FrReflI + FrRefl) / (1 + FrReflI * FrRefl);
  result += complex_norm(FrRefl);

  return result / 2;
}

float film_refl(py::array_t<float> w_i_numpy, py::array_t<float> w_o_numpy, float thickness, float lambda) {
  py::buffer_info w_i_info = w_i_numpy.request();
  float3* w_i = static_cast<float3*>(w_i_info.ptr);

  float cosThetaI = LiteMath::normalize(*w_i).z;

  // hardcode some parameters (to modify in future)
  float result = FrFilmRefl(cosThetaI, complex(1.f, 0.f), complex(2.f, 0.f), complex(1.5f, 0.f), thickness, lambda);

  return result;
}