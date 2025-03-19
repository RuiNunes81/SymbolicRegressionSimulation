import numpy as np
from KDEpy import FFTKDE
import random as rd
from AuxiliaryFunctions import *

class KernelDensities:

    def __init__(self, grid_size):
        self.GridSize = grid_size

    def estimate(self, data, kernel="gaussian", bw_rule="silverman"):
        """
        This method estimate kernel densities and quantile functions 
        for both Distributions and Symmetric Distribution. It will 
        apply KDE.

        **Parameters**
        ---------------------
        data: Dataframe
            This parameter will have the aggregated data that we
            intend to estimate.
        kernel: string
            Name fo the kernel function, that we want to apply in 
            the KDE method.
            Kernel available:
            ["gaussian", "exponential", "box", "tri", "epa", 
            "biweight", "triweight", "tricube", "cosine"]
        bw_rule: string    
            Bandwidth rule to apply in the KDE method.
            Available options: ["silverman", "ISJ"]
        
        Available propeties:

        KernelPoints : points where function were estimated
        KernelValues : KDE estimation
        KernelSymmetricPoints : points where symmetric distribution function were estimated
        KernelSymmetricValues : KDE for the symmetric distribution estimation
        CDFValues : Cummulative density estimation
        QFPoints : points where quantile function were estimated
        QFValues : Quantile function estimation
        Cdf_Symmetric_Values : Cummulative for symmetric distribution density estimation
        QFSymmetricValues : Symmetric distribution Quantile function estimation


        """
        self.data = data
        self.bw_rule=bw_rule
        self.symetric_data=[-1*xi for xi in self.data]

        self.KernelPoints = np.linspace(min(self.data)-7, max(self.data)+7, num=self.GridSize)
        self.KernelSymmetricPoints = np.linspace(min(self.symetric_data)-7, max(self.symetric_data)+7, num=self.GridSize)
        
        self.KernelValues = FFTKDE(kernel=kernel, bw=self.bw_rule).fit(self.data).evaluate(self.KernelPoints) 
        self.KernelSymmetricValues = FFTKDE(kernel=kernel, bw=self.bw_rule).fit(self.symetric_data).evaluate(self.KernelSymmetricPoints) 

        self.CDFValues = np.cumsum(self.KernelValues)/np.cumsum(self.KernelValues)[-1]
        self.QFPoints = get_points_with_final(0.00001, 1, self.GridSize) 
        self.QFValues = [self.KernelPoints[max(np.where(self.CDFValues<=pts)[0])] for pts in self.QFPoints]
 
        self.Cdf_Symmetric_Values = np.cumsum(self.KernelSymmetricValues)/np.cumsum(self.KernelSymmetricValues)[-1]  
        self.QFSymmetricValues = [self.KernelSymmetricValues[max(np.where(self.Cdf_Symmetric_Values<=pts)[0])] for pts in self.QFPoints]

    def get_quantiles(self, data, kernel="gaussian", bw_rule="silverman"):
        self.estimate(data, kernel, bw_rule)
        return [self.QFValues, self.QFSymmetricValues ]
    
    def get_quantile(self, data, kernel="gaussian", bw_rule="silverman"):
        self.estimate(data, kernel, bw_rule)
        return self.QFValues if rd.random()<0.5 else self.QFSymmetricValues 
    
    def get_densities(self, data, kernel="gaussian", bw_rule="silverman"):
        self.estimate(data, kernel, bw_rule)
        return [self.KernelPoints, self.KernelValues,self.KernelSymmetricPoints, self.KernelSymmetricValues ]