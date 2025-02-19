# Copyright (c) 2019, Christoph Peters
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the Karlsruhe Institute of Technology nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


"""
This file implements techniques in the paper "Using Moments to Represent 
Bounded Signals for Spectral Rendering" in Python using NumPy (and matplotlib 
to visualize what is happening). The implementation is intentionally 
unoptimized but written to be easy to read and match to formulas and 
recommendations in the paper.
"""
import numpy as np
from matplotlib import pyplot

def WavelengthToPhase(Wavelength,WavelengthMin,WavelengthMax,MirrorSignal,UseWarp):
    """This function implements the various methods to map a wavelength in 
       nanometers to a phase between -pi and pi or -pi and 0 discussed in 
       Section 4.1 of the paper.
      \param Wavelength The wavelength of light in nanometers (e.g. 360 to 830 
             for visible light). May be an array.
      \param WavelengthMin,WavelengthMax The minimal and maximal wavelengths 
             that may be passed through Wavelength. Irrelevant if a warp is 
             used.
      \param MirrorSignal Pass True to obtain phases between -pi and 0 only. 
             This also implies that a different warp is used.
      \param UseWarp Pass False to use a linear mapping and True to use a 
             piecewise linear mapping that is optimized to make Fourier 
             coefficients resemble XYZ coefficients.
      \return An array of same shape as Wavelength with entries between -pi and 
              pi if MirrorSignal is False or -pi and 0 otherwise."""
    if(UseWarp):
        WarpWavelengths=np.linspace(360.0,830.0,95);
        if(MirrorSignal):
            WarpPhases=[-3.141592654,-3.141592654,-3.141592654,-3.141592654,-3.141591857,-3.141590597,-3.141590237,-3.141432053,-3.140119041,-3.137863071,-3.133438967,-3.123406739,-3.106095749,-3.073470612,-3.024748900,-2.963566246,-2.894461907,-2.819659701,-2.741784136,-2.660533432,-2.576526605,-2.490368187,-2.407962868,-2.334138406,-2.269339880,-2.213127747,-2.162806279,-2.114787412,-2.065873394,-2.012511127,-1.952877310,-1.886377224,-1.813129945,-1.735366957,-1.655108108,-1.573400329,-1.490781436,-1.407519056,-1.323814008,-1.239721795,-1.155352390,-1.071041833,-0.986956525,-0.903007113,-0.819061538,-0.735505101,-0.653346027,-0.573896987,-0.498725202,-0.428534515,-0.363884284,-0.304967687,-0.251925536,-0.205301867,-0.165356255,-0.131442191,-0.102998719,-0.079687644,-0.061092401,-0.046554594,-0.035419229,-0.027113640,-0.021085743,-0.016716885,-0.013468661,-0.011125245,-0.009497032,-0.008356318,-0.007571826,-0.006902676,-0.006366945,-0.005918355,-0.005533442,-0.005193920,-0.004886397,-0.004601975,-0.004334090,-0.004077698,-0.003829183,-0.003585923,-0.003346286,-0.003109231,-0.002873996,-0.002640047,-0.002406990,-0.002174598,-0.001942639,-0.001711031,-0.001479624,-0.001248405,-0.001017282,-0.000786134,-0.000557770,-0.000332262,0.000000000];
        else:
            WarpPhases=[-3.141592654,-3.130798150,-3.120409385,-3.109941320,-3.099293828,-3.088412485,-3.077521262,-3.065536918,-3.051316899,-3.034645062,-3.011566128,-2.977418908,-2.923394305,-2.836724273,-2.718861152,-2.578040247,-2.424210212,-2.263749529,-2.101393442,-1.939765501,-1.783840979,-1.640026237,-1.516709057,-1.412020736,-1.321787834,-1.241163202,-1.164633877,-1.087402539,-1.005125823,-0.913238812,-0.809194293,-0.691182886,-0.559164417,-0.416604255,-0.267950333,-0.117103760,0.033240425,0.181302905,0.326198172,0.468115940,0.607909250,0.746661762,0.885780284,1.026618635,1.170211067,1.316774725,1.465751396,1.615788104,1.764019821,1.907870188,2.044149116,2.170497529,2.285044356,2.385874947,2.472481955,2.546360090,2.608833897,2.660939270,2.703842819,2.739031973,2.767857527,2.791481523,2.810988348,2.827540180,2.842073425,2.855038660,2.866280238,2.876993433,2.887283163,2.897272230,2.907046247,2.916664580,2.926170139,2.935595142,2.944979249,2.954494971,2.964123227,2.973828479,2.983585490,2.993377472,3.003193861,3.013027335,3.022872815,3.032726596,3.042586022,3.052449448,3.062315519,3.072183298,3.082052181,3.091921757,3.101791675,3.111661747,3.121531809,3.131398522,3.141592654];
        return np.interp(Wavelength,WarpWavelengths,WarpPhases);
    else:
        NormalizedWavelengths=(Wavelength-WavelengthMin)/(WavelengthMax-WavelengthMin);
        return ((1.0 if MirrorSignal else 2.0)*np.pi)*NormalizedWavelengths-np.pi;


def ComputeTrigonometricMoments(Phase,Signal,nMoment,MirrorSignal):
    """This function computes trigonometric moments for the given 2 pi periodic 
       signal. If the signal is bounded between zero and one, they will be 
       bounded trigonometric moments. Phase has to be a sorted array, Signal 
       must have same shape to provide corresponding values. The function uses 
       linear interpolation between the sample points. The values of the first 
       and last sample point are repeated towards the end of the domain. If 
       MirrorSignal is True, the signal is mirrored at zero, which means that 
       the moments are real.
      \return An array of shape (nMoment+1,) where entry j is the j-th 
              trigonometric moment."""
    # We do not want to treat out of range values as zero but want to continue 
    # the outermost value to the end of the integration range
    Phase=np.concatenate([[-np.pi],Phase,[0.0 if MirrorSignal else np.pi]]);
    Signal=np.concatenate([[Signal[0]],Signal,[Signal[-1]]]);
    # Now we are ready to integrate
    nSample=Phase.size;
    TrigonometricMoments=np.zeros((nMoment+1,),dtype=complex);
    for l in range(nSample-1):
        if(Phase[l]>=Phase[l+1]):
            continue;
        Gradient=(Signal[l+1]-Signal[l])/(Phase[l+1]-Phase[l]);
        YIntercept=Signal[l]-Gradient*Phase[l];
        for j in range(1,nMoment+1):
            CommonSummands=Gradient*1.0/j**2+YIntercept*1.0j/j;
            TrigonometricMoments[j]+=(CommonSummands+Gradient*1.0j*j*Phase[l+1]/j**2)*np.exp(-1.0j*j*Phase[l+1]);
            TrigonometricMoments[j]-=(CommonSummands+Gradient*1.0j*j*Phase[l  ]/j**2)*np.exp(-1.0j*j*Phase[l]);
        TrigonometricMoments[0]+=0.5*Gradient*Phase[l+1]**2+YIntercept*Phase[l+1];
        TrigonometricMoments[0]-=0.5*Gradient*Phase[l]**2+YIntercept*Phase[l];
    TrigonometricMoments*=0.5/np.pi;
    if(MirrorSignal):
        return 2.0*TrigonometricMoments.real;
    else:
        return TrigonometricMoments;


def BoundedTrigonometricMomentsToExponentialMoments(BoundedTrigonometricMoments):
    """This function applies the recurrence specified Proposition 2 to turn 
       bounded trigonometric moments into exponential moments.
      \return An array of same shape as BoundedTrigonometricMoments. Entry 0 
              does not hold the zeroth exponential moment but its complex 
              counterpart. Take its real part times two to get the actual 
              zeroth exponential moment."""
    nMoment=BoundedTrigonometricMoments.size-1;
    ExponentialMoments=np.zeros((nMoment+1,),dtype=complex);
    ExponentialMoments[0]=0.25/np.pi*np.exp(np.pi*1.0j*(BoundedTrigonometricMoments[0]-0.5));
    for l in range(1,nMoment+1):
        for j in range(l):
            ExponentialMoments[l]+=(l-j)*ExponentialMoments[j]*BoundedTrigonometricMoments[l-j];
        ExponentialMoments[l]*=2.0j*np.pi/l;
    return ExponentialMoments;


def LevinsonsAlgorithm(FirstColumn):
    """Implements Levinson's algorithm to solve a special system of linear 
       equations. The matrix has the given first column and is a Hermitian 
       Toeplitz matrix, i.e. it has constant diagonals. The right-hand side 
       vector is the canonical basis vector (1,0,...,0).
      \return A vector of same shape as FirstColumn.
      \note This algorithm provides an efficient way to compute the vector q in 
            the paper, which is called EvaluationPolynomial in this program."""
    Solution=np.zeros_like(FirstColumn);
    Solution[0]=1.0/FirstColumn[0];
    for j in range(1,FirstColumn.shape[0]):
        DotProduct=np.dot(Solution[0:j],FirstColumn[j:0:-1]);
        Solution[0:j+1]=(Solution[0:j+1]-DotProduct*np.conj(Solution[j::-1]))/(1-np.abs(DotProduct)**2);
    return Solution;


def EvaluateHerglotzTransform(PointInDisk,ExponentialMoments,EvaluationPolynomial):
    """This function implements Algorithm 1 from the paper. PointInDisk may be 
       an array. In this case the retur value is an array of same shape."""
    nMoment=ExponentialMoments.size-1;
    CoefficientList=(nMoment+1)*[None];
    CoefficientList[nMoment]=EvaluationPolynomial[0];
    for l in range(nMoment-1,-1,-1):
        CoefficientList[l]=EvaluationPolynomial[nMoment-l]+CoefficientList[l+1]/PointInDisk;
    HerglotzTransform=np.zeros_like(PointInDisk);
    for k in range(1,nMoment+1):
        HerglotzTransform+=CoefficientList[k]*ExponentialMoments[k];
    HerglotzTransform*=2.0/CoefficientList[0];
    HerglotzTransform+=ExponentialMoments[0];
    return HerglotzTransform;


def ComputeLagrangeMultipliers(ExponentialMoments,EvaluationPolynomial):
    """This function implements the definition of Lagrange multipliers from 
       Proposition 6 in the paper. It offers an efficient way to evaluate the 
       bounded MESE in many locations.
      \return An array of same shape as ExponentialMoments."""
    nMoment=EvaluationPolynomial.size-1;
    # Compute the autocorrelation of EvaluationPolyanomial
    Autocorrelation=np.zeros_like(EvaluationPolynomial);
    for k in range(nMoment+1):
        for j in range(nMoment+1-k):
            Autocorrelation[k]+=np.conj(EvaluationPolynomial[j+k])*EvaluationPolynomial[j];
    # Use it to construct the Lagrange multipliers
    LagrangeMultipliers=np.zeros_like(ExponentialMoments);
    for l in range(nMoment+1):
        for k in range(nMoment+1-l):
            LagrangeMultipliers[l]+=ExponentialMoments[k]*Autocorrelation[k+l];
    LagrangeMultipliers/=np.pi*1.0j*EvaluationPolynomial[0];
    return LagrangeMultipliers;


def EvaluateBoundedMESEDirect(Phase,BoundedTrigonometricMoments):
    """This function implements evaluation of the bounded maximum entropy 
       spectral estimate (MESE) using direct computation of the Herglotz 
       transform (through Algorithm 1).
      \return An array of same shape as Phase with values between zero and one 
              sampling a signal that realizes the given bounded trigonometric 
              moments."""
    ExponentialMoments=BoundedTrigonometricMomentsToExponentialMoments(BoundedTrigonometricMoments);
    ToeplitzFirstColumn=ExponentialMoments*0.5/np.pi;
    ToeplitzFirstColumn[0]=2.0*ToeplitzFirstColumn[0].real;
    EvaluationPolynomial=LevinsonsAlgorithm(ToeplitzFirstColumn);
    PointInDisk=np.exp(1.0j*Phase);
    ExponentialMoments[0]*=2.0;
    HerglotzTransform=EvaluateHerglotzTransform(PointInDisk,ExponentialMoments,EvaluationPolynomial);
    return np.angle(HerglotzTransform)/np.pi+0.5;


def EvaluateBoundedMESELagrange(Phase,BoundedTrigonometricMoments):
    """Implements the same interface as EvaluateBoundedMESEDirect() but uses 
       the detour of Lagrange multipliers such that it is more efficient when 
       the bounded MESE is evaluated for many phases."""
    nMoment=BoundedTrigonometricMoments.size-1;
    # Compute the Lagrange multipliers
    ExponentialMoments=BoundedTrigonometricMomentsToExponentialMoments(BoundedTrigonometricMoments);
    ToeplitzFirstColumn=ExponentialMoments*0.5/np.pi;
    ToeplitzFirstColumn[0]=2.0*ToeplitzFirstColumn[0].real;
    EvaluationPolynomial=LevinsonsAlgorithm(ToeplitzFirstColumn);
    LagrangeMultipliers=ComputeLagrangeMultipliers(ExponentialMoments,EvaluationPolynomial);
    # Evaluate the Fourier series (we only need the real part)
    FourierSeries=np.zeros_like(Phase);
    for l in range(1,nMoment+1):
        FourierSeries+=np.real(LagrangeMultipliers[l]*np.exp(-1.0j*l*Phase));
    FourierSeries*=2.0;
    FourierSeries+=LagrangeMultipliers[0].real;
    return np.arctan(FourierSeries)/np.pi+0.5;


def EvaluateMESE(Phase,TrigonometricMoments):
    """This function implements the MESE (without upper bound), i.e. a positive 
       signal realizing the given trigonometric moments."""
    nMoment=TrigonometricMoments.size-1;
    EvaluationPolynomial=LevinsonsAlgorithm(TrigonometricMoments*0.5/np.pi);
    FourierSeries=np.zeros(Phase.shape,dtype=complex);
    for j in range(nMoment+1):
        FourierSeries+=np.conj(EvaluationPolynomial[j])*np.exp(-1.0j*j*Phase)*0.5/np.pi;
    return (EvaluationPolynomial[0].real*0.5/np.pi)/np.abs(FourierSeries)**2;


if(__name__=="__main__"):
    # Choose the representation of the reflectance spectrum and the 
    # reconstruction method
    nMoment=6;
    MirrorSignal=True;
    UseWarp=True;
    UseLagrangeMultipliers=True;
    # This is the reflectance of the yellow_green patch on the color checker
    ReflectanceWavelength=np.linspace(400.0,700.0,61);
    Reflectance=np.asarray([0.060000000,0.061000000,0.061000000,0.061000000,0.062000000,0.063000000,0.064000000,0.066000000,0.068000000,0.071000000,0.075000000,0.079000000,0.085000000,0.093000000,0.104000000,0.118000000,0.135000000,0.157000000,0.185000000,0.221000000,0.269000000,0.326000000,0.384000000,0.440000000,0.484000000,0.516000000,0.534000000,0.542000000,0.545000000,0.541000000,0.533000000,0.524000000,0.513000000,0.501000000,0.487000000,0.472000000,0.454000000,0.436000000,0.416000000,0.394000000,0.374000000,0.358000000,0.346000000,0.337000000,0.331000000,0.328000000,0.325000000,0.322000000,0.320000000,0.319000000,0.319000000,0.320000000,0.324000000,0.330000000,0.337000000,0.345000000,0.354000000,0.362000000,0.368000000,0.375000000,0.379000000]);
    # Compute bounded trigonometric moments
    ReflectancePhase=WavelengthToPhase(ReflectanceWavelength,ReflectanceWavelength.min(),ReflectanceWavelength.max(),MirrorSignal,UseWarp);
    BoundedTrigonometricMoments=ComputeTrigonometricMoments(ReflectancePhase,Reflectance,nMoment,MirrorSignal);
    # Use the bounded MESE for a dense reconstruction of the reflectance 
    # spectrum
    DenseWavelength=np.linspace(ReflectanceWavelength.min(),ReflectanceWavelength.max(),1001);
    DensePhase=WavelengthToPhase(DenseWavelength,ReflectanceWavelength.min(),ReflectanceWavelength.max(),MirrorSignal,UseWarp);
    if(UseLagrangeMultipliers):
        BoundedMESE=EvaluateBoundedMESELagrange(DensePhase,BoundedTrigonometricMoments);
    else:
        BoundedMESE=EvaluateBoundedMESEDirect(DensePhase,BoundedTrigonometricMoments);
    # Plot ground truth and reconstruction
    pyplot.subplot(xlim=(ReflectanceWavelength.min(),ReflectanceWavelength.max()),ylim=(0.0,1.0),xlabel="Wavelength / nm",ylabel="Reflectance");
    pyplot.plot(ReflectanceWavelength,Reflectance,label="Ground truth");
    pyplot.plot(DenseWavelength,BoundedMESE,label="Bounded MESE, m=%d%s%s"%(nMoment,", mirrored" if MirrorSignal else ", complex",", warped" if UseWarp else ""));
    pyplot.legend(frameon=False);
    # Verify that the moments of the reconstruction are correct. Since 
    # DensePhase does not quite cover the whole range of valid phases, we 
    # sample the reconstruction a second time.
    DensePhase=np.linspace(-np.pi,0.0 if MirrorSignal else np.pi,1001);
    if(UseLagrangeMultipliers):
        BoundedMESE=EvaluateBoundedMESELagrange(DensePhase,BoundedTrigonometricMoments);
    else:
        BoundedMESE=EvaluateBoundedMESEDirect(DensePhase,BoundedTrigonometricMoments);
    ReconstructedBoundedTrigonometricMoments=ComputeTrigonometricMoments(DensePhase,BoundedMESE,nMoment,MirrorSignal);
    print("Moment error (bounded MESE): ",np.linalg.norm(ReconstructedBoundedTrigonometricMoments-BoundedTrigonometricMoments));

    # For emission spectra we use more moments and no warp because they tend to 
    # be more complicated
    nMoment=32;
    MirrorSignal=True;
    # This is the emission spectrum of a fluorescent lamp
    EmissionWavelength=np.linspace(312.5,811.5,999);
    Emission=np.asarray([0.03183,0.02949,0.02075,0.00807,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00281,0.00439,0.00257,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00005,0.00032,0.00027,0.00060,0.00142,0.00190,0.00558,0.00773,0.00783,0.00437,0.00076,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00047,0.00176,0.00185,0.00095,0.00000,0.00000,0.00266,0.00169,0.00037,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00090,0.00475,0.00823,0.00661,0.00546,0.00271,0.00265,0.00472,0.00521,0.00544,0.00392,0.00160,0.00000,0.00000,0.00000,0.00000,0.00000,0.00536,0.02746,0.06113,0.09360,0.11145,0.10533,0.07810,0.05022,0.02810,0.01515,0.00646,0.00001,0.00000,0.00000,0.00017,0.00205,0.00191,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00111,0.00480,0.00682,0.00464,0.00110,0.00000,0.00002,0.00029,0.00020,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00024,0.00000,0.00022,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00002,0.00118,0.00329,0.00520,0.00666,0.00434,0.00184,0.00179,0.00059,0.00115,0.00195,0.00340,0.00471,0.01317,0.04988,0.12346,0.20524,0.24420,0.22047,0.14463,0.07435,0.03571,0.02632,0.02591,0.02307,0.01642,0.00911,0.00342,0.00165,0.00169,0.00268,0.00328,0.00329,0.00192,0.00317,0.00422,0.00453,0.00546,0.00413,0.00508,0.00678,0.00753,0.00674,0.00624,0.00508,0.00509,0.00593,0.00637,0.00656,0.00608,0.00531,0.00467,0.00528,0.00658,0.00730,0.00797,0.00652,0.00652,0.00885,0.01169,0.01330,0.01241,0.00971,0.00920,0.01072,0.01075,0.01258,0.01346,0.01516,0.01670,0.01500,0.01622,0.02043,0.02490,0.02979,0.05001,0.13486,0.36258,0.66126,0.87177,0.83723,0.60594,0.32710,0.13781,0.06077,0.03594,0.02657,0.02265,0.01907,0.01775,0.01916,0.02089,0.02335,0.02384,0.02206,0.01974,0.01775,0.01984,0.02336,0.02659,0.02621,0.02557,0.02689,0.02814,0.02638,0.02050,0.01707,0.01526,0.01758,0.01839,0.01852,0.01807,0.01876,0.02014,0.02036,0.02159,0.01983,0.01828,0.01855,0.01811,0.01768,0.01646,0.01450,0.01543,0.01747,0.01784,0.01792,0.01730,0.01729,0.01780,0.01835,0.01960,0.01813,0.01523,0.01225,0.01116,0.01358,0.01481,0.01464,0.01427,0.01450,0.01595,0.01605,0.01544,0.01418,0.01303,0.01319,0.01296,0.01464,0.01539,0.01576,0.01787,0.02032,0.01991,0.01901,0.01778,0.01765,0.01899,0.01974,0.01968,0.02081,0.02057,0.01812,0.01486,0.01408,0.01631,0.01952,0.02394,0.02725,0.03111,0.03460,0.03862,0.04397,0.05210,0.06303,0.07581,0.09005,0.10531,0.12224,0.13913,0.15221,0.15977,0.16024,0.15716,0.15020,0.14103,0.13206,0.12378,0.11772,0.11313,0.11063,0.10707,0.10022,0.09238,0.08555,0.08027,0.07729,0.07392,0.07113,0.06808,0.06391,0.06047,0.05592,0.05356,0.04986,0.04574,0.04123,0.03665,0.03348,0.03017,0.02693,0.02394,0.02124,0.01992,0.01856,0.01863,0.01795,0.01799,0.01805,0.01815,0.01741,0.01693,0.01699,0.01712,0.01586,0.01384,0.01203,0.01213,0.01416,0.01523,0.01541,0.01526,0.01498,0.01456,0.01330,0.01246,0.01241,0.01309,0.01443,0.01411,0.01512,0.01543,0.01611,0.01622,0.01485,0.01237,0.01089,0.01029,0.01168,0.01362,0.01359,0.01260,0.01080,0.01076,0.01122,0.01149,0.01112,0.01209,0.01326,0.01394,0.01389,0.01420,0.01471,0.01493,0.01390,0.01315,0.01370,0.01517,0.01632,0.01684,0.01749,0.02101,0.02619,0.03273,0.03792,0.04030,0.04103,0.04012,0.04190,0.04794,0.05849,0.07639,0.09780,0.12232,0.14726,0.17818,0.22662,0.30577,0.43080,0.59592,0.76055,0.87011,0.88186,0.82737,0.76166,0.72346,0.74437,0.85165,1.06570,1.28210,1.32660,1.17590,0.90671,0.67591,0.54440,0.46602,0.40838,0.35627,0.30960,0.26661,0.22940,0.19965,0.17487,0.15402,0.13470,0.11625,0.09974,0.08546,0.07338,0.06383,0.05677,0.05193,0.04969,0.04491,0.03973,0.03454,0.03274,0.03187,0.03158,0.02883,0.02584,0.02381,0.02323,0.02310,0.02224,0.02066,0.02031,0.02031,0.02064,0.01942,0.01802,0.01606,0.01564,0.01538,0.01521,0.01488,0.01492,0.01464,0.01369,0.01246,0.01223,0.01135,0.01102,0.01003,0.01086,0.01199,0.01359,0.01600,0.02732,0.05904,0.11278,0.16184,0.18317,0.17462,0.17225,0.19120,0.21025,0.20751,0.18799,0.16706,0.15800,0.16107,0.17413,0.18906,0.19824,0.20043,0.19760,0.19154,0.18733,0.19436,0.22298,0.26762,0.30149,0.29477,0.24851,0.18934,0.14842,0.12798,0.12800,0.13592,0.15307,0.17962,0.21513,0.25133,0.27003,0.25396,0.21147,0.16101,0.12577,0.10811,0.10260,0.10423,0.10871,0.12116,0.13824,0.16119,0.18220,0.18892,0.17842,0.15230,0.12370,0.10054,0.08511,0.07577,0.07104,0.07128,0.07507,0.07836,0.08222,0.08907,0.10029,0.11541,0.13450,0.16321,0.20835,0.28525,0.42854,0.74318,1.28260,1.98160,2.50960,2.60340,2.22100,1.68590,1.25740,1.00590,0.86514,0.75714,0.64514,0.52353,0.41218,0.32364,0.26156,0.21912,0.19072,0.17270,0.16387,0.16293,0.16424,0.16667,0.16742,0.17001,0.17361,0.18034,0.18947,0.19862,0.20592,0.21157,0.21577,0.22009,0.22143,0.21836,0.21096,0.20303,0.19807,0.19681,0.20724,0.23761,0.28877,0.33451,0.34214,0.29759,0.22037,0.14536,0.09589,0.06834,0.05638,0.04923,0.04675,0.04636,0.04507,0.04072,0.03586,0.03298,0.03357,0.03433,0.03388,0.03239,0.02874,0.02497,0.02097,0.02374,0.02725,0.02870,0.02767,0.02611,0.02723,0.02679,0.02596,0.02709,0.02802,0.02793,0.02496,0.02355,0.02643,0.03300,0.04684,0.06912,0.09304,0.10242,0.09417,0.07694,0.06300,0.05589,0.05033,0.04560,0.03827,0.03365,0.02854,0.02634,0.02628,0.02984,0.03226,0.03216,0.02992,0.02733,0.02439,0.02408,0.02443,0.02613,0.03074,0.04197,0.05554,0.06506,0.05996,0.04694,0.03306,0.02773,0.02519,0.02387,0.02143,0.01990,0.01292,0.00916,0.00956,0.01480,0.01783,0.01691,0.01702,0.01404,0.01028,0.00776,0.01151,0.01353,0.01262,0.00752,0.00772,0.00663,0.00480,0.00452,0.00605,0.00982,0.00942,0.00863,0.00951,0.01099,0.01282,0.01298,0.01573,0.01831,0.01877,0.01796,0.01318,0.00856,0.00709,0.01268,0.01888,0.02218,0.02146,0.02032,0.02244,0.03325,0.04689,0.05390,0.05241,0.04452,0.03212,0.01950,0.01029,0.00849,0.00685,0.00676,0.00815,0.01425,0.02679,0.03881,0.04250,0.03391,0.01795,0.00639,0.00254,0.00522,0.00578,0.00281,0.00134,0.00004,0.00054,0.00135,0.00267,0.00420,0.00272,0.00717,0.01148,0.01327,0.01113,0.01344,0.02051,0.02905,0.04216,0.06547,0.10035,0.12991,0.14735,0.14673,0.14957,0.15497,0.15811,0.13606,0.10156,0.07295,0.07317,0.09199,0.11486,0.12240,0.10485,0.07150,0.04066,0.02214,0.01333,0.00693,0.00168,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00083,0.00621,0.00822,0.00297,0.00053,0.00000,0.00000,0.00000,0.00069,0.00016,0.00000,0.00000,0.00196,0.00421,0.00059,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00125,0.00026,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00202,0.02020,0.02191,0.01320,0.00336,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00249,0.00022,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00138,0.00000,0.00000,0.00000,0.00572,0.00518,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00000,0.00119,0.01094,0.01152,0.01058,0.02523,0.04864]);
    # Compute trigonometric moments
    EmissionPhase=WavelengthToPhase(EmissionWavelength,EmissionWavelength.min(),EmissionWavelength.max(),MirrorSignal,False);
    TrigonometricMoments=ComputeTrigonometricMoments(EmissionPhase,Emission,nMoment,MirrorSignal);
    # Reconstruct using the MESE
    DenseWavelength=np.linspace(EmissionWavelength.min(),EmissionWavelength.max(),5001);
    DensePhase=WavelengthToPhase(DenseWavelength,EmissionWavelength.min(),EmissionWavelength.max(),MirrorSignal,False);
    MESE=EvaluateMESE(DensePhase,TrigonometricMoments);
    # Plot ground truth and reconstruction
    pyplot.figure();
    pyplot.subplot(xlim=(EmissionWavelength.min(),EmissionWavelength.max()),ylim=(0.0,1.1*Emission.max()),xlabel="Wavelength / nm",ylabel="Emission");
    pyplot.plot(EmissionWavelength,Emission,label="Ground truth");
    pyplot.plot(DenseWavelength,MESE,label="MESE, m=%d%s"%(nMoment,", mirrored" if MirrorSignal else ", complex"));
    pyplot.legend(frameon=False);
    # Verify that the moments of the reconstruction are correct
    ReconstructedTrigonometricMoments=ComputeTrigonometricMoments(DensePhase,MESE,nMoment,MirrorSignal);
    print("Moment error (MESE): ",np.linalg.norm(ReconstructedTrigonometricMoments-TrigonometricMoments));
    pyplot.show();
