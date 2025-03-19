import numpy as np
from scipy.stats import norm
from bisect import bisect
from KDEpy import FFTKDE
import random as rd
from AuxiliaryFunctions import *

class KernelDensities:
    GridSize = 1000
    data = None
    N = 0
    KernelPoints = []
    KernelValues = []
    KernelSymmetricPoints = []
    KernelSymmetricValues = []
    CDFValues = []
    Cdf_Symmetric_Values = []
    QFPoints = []
    QFValues = []
    QFSymmetricValues = []

    def __init__(self, grid_size,threshold=0.5):
        self.GridSize = grid_size
        self.threshold = threshold

    def estimate(self, data, kernel="gaussian", bw_rule="silverman", withSymmetric=True):
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

        self.KernelPoints, self.KernelValues = self.get_kde(kernel)
        self.KernelSymmetricPoints, self.KernelSymmetricValues = self.get_kde(kernel,True)
        self.CDFValues = self.get_cdf()
        self.QFPoints, self.QFValues = self.get_qf()
        if withSymmetric:
            self.Cdf_Symmetric_Values = self.get_cdf_symmetric()    
            self.QFSymmetricValues = self.get_qf_symmetric()

    def get_kde(self,kernel, is_symmetric=False):
        if is_symmetric:
            data=[-1*xi for xi in self.data]
            x,y=FFTKDE(kernel=kernel, bw=self.bw_rule).fit(data).evaluate(self.GridSize)
        else:
            x,y=FFTKDE(kernel=kernel, bw=self.bw_rule).fit(self.data).evaluate(self.GridSize)
        return x,y

    def get_cdf(self):
        cdf_x = np.zeros(self.GridSize)
        cdf_x[0] = 0
        c = 1
        xMin=np.min(self.KernelPoints)
        for i in range(1, len(self.KernelPoints)):
            delta = (self.KernelPoints[i] - xMin) / i
            y = self.KernelValues[0:i]
            k = (y[0] + y[i - 1]) / 2
            p = sum(y[1:(i - 1)])
            cdf_x[c] = delta * (k + p)
            c = c + 1

        return cdf_x

    def get_cdf_symmetric(self):
        cdf_x = np.zeros(self.GridSize)
        cdf_x[0] = 0
        c = 1
        xMax=np.max(self.KernelPoints)
        for i in range(1, len(self.KernelSymmetricPoints)):
            delta = (self.KernelSymmetricPoints[i] + xMax) / i
            y = self.KernelSymmetricValues[0:i]
            k = (y[0] + y[i - 1]) / 2
            p = sum(y[1:(i - 1)])
            cdf_x[c] = delta * (k + p)
            c = c + 1

        return cdf_x

    # function to determinate the quantile

    def get_qf(self):
        points_QF = get_points_with_final(0, 1, self.GridSize)
        QF = np.zeros(self.GridSize)
        cdf = self.get_cdf()
        c = 0
        for p in points_QF:
            h = cdf <= p
            x_index = max(np.where(h)[0])
             
            QF[c] = self.KernelPoints[x_index]
            c += 1

        return points_QF, QF

    def get_qf_symmetric(self):
        points_QF = get_points_with_final(0.001, 1, self.GridSize)
        QF = np.zeros(self.GridSize)
        cdf = self.get_cdf_symmetric()
        c = 0
        for p in points_QF:
            h = cdf <= p
            x_index = max(np.where(h)[0])
            QF[c] = self.KernelSymmetricPoints[x_index]
            c = c + 1

        return QF

    def get_cdf_external(self, external_values):
        cdf_x = np.zeros(self.GridSize)
        cdf_x[0] = 0
        c = 1
        xMin=np.min(self.KernelPoints)
        for i in range(1, len(self.KernelPoints)):
            delta = (self.KernelPoints[i] - xMin) / i
            y = external_values[0:i]
            k = (y[0] + y[i - 1]) / 2
            p = sum(y[1:(i - 1)])
            cdf_x[c] = delta * (k + p)
            c = c + 1

        return cdf_x
    
    def get_qf_external(self, external_values):
        points_QF = get_points_with_final(0.001, 1, self.GridSize)
        QF = np.zeros(self.GridSize)
        cdf = self.get_cdf_external(external_values)
        c = 0
        for p in points_QF:
            h = cdf <= p
            x_index = max(np.where(h)[0])
             
            QF[c] = self.KernelPoints[x_index]
            c += 1

        return points_QF, QF

    def get_quantiles(self, data, kernel="gaussian", bw_rule="silverman"):
        self.estimate(data, kernel, bw_rule)
        return [self.QFValues,self.QFSymmetricValues ]
    
    def get_quantile(self, data,is_symetric, kernel="gaussian", bw_rule="silverman"):
        self.estimate(data, kernel, bw_rule,False)
        return self.QFSymmetricValues if is_symetric==1 else self.QFValues
    
    def get_densities(self, data, kernel="gaussian", bw_rule="silverman"):
        self.estimate(data, kernel, bw_rule)
        return [self.KernelPoints, self.KernelValues,self.KernelSymmetricPoints, self.KernelSymmetricValues ]