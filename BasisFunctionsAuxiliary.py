import numpy as np
import pandas as pd
from scipy import integrate
import BSplines as bs
from scipy import special


class BasisFunctionsAuxiliary:
    grid_size = 1000
    basis_type = "Fourier"
    points = []
    b_spline_degree = 7
    b_spline_knots = []
    b_spline_basis = []
    fourier_w = 1
    fourier_tau = 1
    n_basis = 20

    def __init__(self, _grid_size, _basis_type, x_points, n_basis):
        self.grid_size = _grid_size
        self.basis_type = _basis_type
        self.points = x_points
        self.n_basis = n_basis

    def update_n_basis(self, n_basis):
        self.n_basis = n_basis

    def initiate_b_splines(self, spline_degree, spline_knots):
        self.b_spline_degree = spline_degree
        self.b_spline_knots = spline_knots

    def initiate_fourier(self, fourier_weight, fourier_tau):
        self.fourier_w = fourier_weight
        self.fourier_tau = fourier_tau

    def approx_function(self, n, fourier_is_sin):
        if self.basis_type == "Fourier":
            return self.fourier(n, fourier_is_sin)
        elif self.basis_type == "Legendre":
            return self.legendre_polynomial(n)
        elif self.basis_type == "Pol":
            return self.polynomial(n)
        elif self.basis_type == "BSpline":
            return self.b_spline(n)
        elif self.basis_type == "soft":
            return self.soft_threshold(self.points, n)
        elif self.basis_type == "Exponential":
            return self.exponential(n)        
        else:
            return np.zeros(len(self.points))

    def exponential(self, n):
        return np.exp(self.points * n/2)
    
    def exponential(self, n):
        return np.exp(self.points * n/2)

    def fourier(self, n, p):
        if n == 0:
            return np.ones(len(self.points)) / np.sqrt(self.fourier_tau)
        elif p == 0:
            return np.sin(n * self.fourier_w * self.points) / np.sqrt(self.fourier_tau / 2)
        else:
            return np.cos(n * self.fourier_w * self.points) / np.sqrt(self.fourier_tau / 2)
        
    def polynomial(self, n):
        return self.points ** n
    
    def legendre_polynomial(self, n):
        x = self.points
        p_monic=special.hermite(n, monic=False) 
        x = np.linspace(0, 1, 400)
        return p_monic(x)
    
    def soft_threshold(self, i):
        x = self.points
        return 1 / (1 + np.exp(-x.T * i))
    
    def get_b_splines(self):
        ol = bs.BSplines(self.b_spline_knots, self.b_spline_degree, self.grid_size)
        self.b_spline_basis = ol.get_b_spline_functions()
        return self.b_spline_basis

    def get_d_b_splines(self, order):
        ol = bs.BSplines(self.b_spline_knots, self.b_spline_degree, self.grid_size)
        self.b_spline_basis = ol.diff(order)
        return self.b_spline_basis
    
    
    def function_estimation(self, c):
        val = 0
        if self.basis_type == "Fourier" or self.basis_type == "Fourier1":
            f = int((self.n_basis - 1) / 2) + 1
            val += c[0] * self.approx_function(0, 0)
            j = 1
            for r in range(1, f):
                val += c[j] * self.approx_function(r, 0)
                j = j + 1
                val += c[j] * self.approx_function(r, 1)
                j += 1
        elif self.basis_type == "BSpline":
            for r in range(self.n_basis):
                val += c[r] * self.b_spline_basis[r]
        else:
            for r in range(self.n_basis):
                val += c[r] * self.approx_function(r, 0)

        return val

    def calculate_mi(self, i, j, df_w, df_z):
        cols = self.n_basis
        data = {}
        for q in range(self.n_basis):
            data["%s" % q] = np.zeros(cols)

        df_zeros = pd.DataFrame(data=data)
        M = df_zeros.to_numpy().T

        ww = df_w["w_%s%s" % (i, j)]
        for p in range(self.n_basis):
            for k in range(self.n_basis):
                M[p, k] = integrate.simpson(ww * df_z["phi_%s_%s" % (p, k)], x=self.points)
        return M

    def calculate_r(self, i, li, df_R):
        cols = self.n_basis
        data = {}
        for q in range(self.n_basis):
            data["%s" % q] = np.zeros(cols)

        df_zeros = pd.DataFrame(data=data)
        M = df_zeros.to_numpy().T

        if i == 0:
            return M
        else:
            for p in range(self.n_basis):
                for k in range(self.n_basis):
                    M[p, k] = li * integrate.simpson(
                        df_R["phi_%s_%s" % (p, p)] * df_R["phi_%s_%s" % (p, k)], x=self.points)
            return M

    def calculate_mi_xy(self, i, df_xy, df_basis):
        cols = self.n_basis
        c = np.zeros(cols)
        ww = df_xy["xy_%s" % i]
        for p in range(self.n_basis):
            c[p] = integrate.simpson(ww * df_basis["phi_%s" % p], x=self.points)

        return c