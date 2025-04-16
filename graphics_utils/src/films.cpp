#include "films.hpp"
#include "common.hpp"

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

py::array_t<float> film_refl(py::array_t<float> w_i_numpy, py::array_t<float> w_o_numpy, py::array_t<float> etaI_numpy, py::array_t<float> etaF_numpy, py::array_t<float>  etaT_numpy, float thickness, py::array_t<float> lambdas) 
{
  py::buffer_info w_i_info = w_i_numpy.request();
  float3* w_i = static_cast<float3*>(w_i_info.ptr);

  py::buffer_info etaI_info = etaI_numpy.request();
  float* etaI = static_cast<float*>(etaI_info.ptr);

  py::buffer_info etaF_info = etaF_numpy.request();
  float* etaF = static_cast<float*>(etaF_info.ptr);

  py::buffer_info etaT_info = etaT_numpy.request();
  float* etaT = static_cast<float*>(etaT_info.ptr);

  float cosThetaI = LiteMath::normalize(*w_i).z;

  py::buffer_info lambdas_info = lambdas.request();
  float* lambdasT = static_cast<float*>(lambdas_info.ptr);

  if (lambdas_info.ndim != 1)
    throw std::runtime_error("Number of dimensions must be one");

  py::array_t<float> result(lambdas_info.size);
  auto result_buf = result.request();
  float* resultT = static_cast<float*>(result_buf.ptr);

  for(int i = 0; i < lambdas_info.size; ++i)
    resultT[i] = FrFilmRefl(cosThetaI, complex(etaI[0], etaI[1]), complex(etaF[0], etaF[1]), complex(etaT[0], etaT[1]), thickness, lambdasT[i]);

  return result;
}