import random as rd
import KernelDensities as Kde
import cvxpy as cp
import matplotlib.pyplot as plt
from numpy import *

import BasisFunctionsAuxiliary as Auxiliary
from AuxiliaryFunctions import *


class SymbolicDensityRegression:
    """This Class deals with linear models (both concurrent FLM as DSD model).
    It will also generate KDE estimations if needed to raw aggregated data.
    If Loading previous generated data please beware that should follow the format:
    Each variable must be represented in the dataframe by a array with dimension 2,
    for example:
        

    Returns:
        _type_: _description_
    """

    ##############################
    # Start - Initialize Functions
    ##############################
    def InitializeVariables(self, grid_size, data_range):
        self.GridSize = grid_size
        self.DataRange = data_range
        points = get_points(data_range[0], data_range[1], self.GridSize)
        self.UsedPoints = points
        self.data = None
        self.N = 0
        self.q = 0
        self.q_dsd = 0
        self.n_bases = 21
        self.BasisDegreeOfSpline = 1
        self.NumberOfNonDecreasingResponses = 0
        self.SymmetricUsedPoints = None
        self.KdeKernelPoints = None
        self.Regression_Coefficients = []
        self.df_variables = None
        self.co_variables = []
        self.response = ""
        self.response_postfix = ""
        self.BasisType = "BSpline"
        self.variable_column = ""
        self.variable_description_column = ""
        self.Variables = []
        self.Basis_df = None
        self.kernel_df = None
        self.quantile_df = None
        self.Weighted_df = None
        self.Xy_df = None
        self.Phi_by_Phi_df = None
        self.auxiliar_methods = None
        self.QP_sol = None
        self.Q_Matrix = None
        self.r_Matrix = None
        self.dsd_y_hat = None
        self.fda_y_hat = None
        self.Q_sums_Matrix = None
        self.r_sums_Matrix = None
        self.df_beta = None
        self.quantile_clr_df = None
        self.K_value = 0

        self.fda_d_means=0
        self.dsd_d_means=0

        self.l1_sse = []
        self.l2_sse = []
        self.elastic_sse = []
        self.l1_mse = []
        self.l2_mse = []
        self.elastic_mse = []
        self.l1_mse_c = []
        self.l2_mse_c = []
        self.elastic_mse_c = []
        self.l1_l = []
        self.l2_l = []
        self.elastic_l = []
        self.l1_b = []
        self.l2_b = []
        self.elastic_b = []
        self.elastic_alpha = []
        self.unp_test_mse = -1
        self.unp_train_mse = -1
        self.unp_b = []
        self.unp_fda_sse = []
        self.unp_fda_mse = -1
        self.unp_fda_mse_test = -1
        self.unp_fda_y_hat = None
        self.dsd_l1_y_hat = None
        self.dsd_l2_y_hat = None
        self.dsd_elastic_y_hat = None
        self.unp_dsd_y_hat = None
        self.is_k_fold = False
        self.test_variable = []
        self.train_variables = []
        self.df_fda_y_hat = None
        self.df_dsd_un_yhat = None
        self.unp_dsd_y_beta = []
        self.k_fold = 5
        self.k_folds = []
        self.num_repeats_k_fold = 5
        self.unp_fda_train_mse = -1
        self.unp_fda_test_mse = -1
        self.mean_response = []
        self.error_observe_to_mean = 0
        self.fda_r_2_train = 0
        self.fda_r_2_test = 0
        self.dsd_r2_train = 0
        self.dsd_r2_test = 0

        self.cv_train = []
        self.cv_test = []
        self.cv_r2_train = 0
        self.cv_r2_test = 0

        self.dsd_r2_train_l1 = 0
        self.dsd_r2_test_l1 = 0
        self.dsd_r2_train_l2 = 0
        self.dsd_r2_test_l2 = 0
        self.dsd_r2_train_elastic = 0
        self.dsd_r2_test_elastic = 0
        self.l1_mse_test = 0
        self.l2_mse_test = 0
        self.elastic_mse_test = 0
        self.dsd_r2_train_l1_c = 0
        self.dsd_r2_test_l1_c = 0
        self.dsd_r2_train_l2_c = 0
        self.dsd_r2_test_l2_c = 0
        self.dsd_r2_train_elastic_c = 0
        self.dsd_r2_test_elastic_c = 0
        self.l1_mse_test_c = 0
        self.l2_mse_test_c = 0
        self.elastic_mse_test_c = 0
        
        self.l1_c_betas = []
        self.l1_c_lambda_values = []
        self.l1_c_mse = []
        self.l1_betas = []
        self.l1_c_betas = []
        self.l2_betas = []
        self.l2_c_betas = []
        self.el_betas = []
        self.el_c_betas = []
        self.l1_lambda_values = []
        self.l1_mse = []

    def __init__(self, grid_size, data_range):
        self.InitializeVariables(grid_size, data_range)
        self.num_repeats_k_fold = 5
        points = get_points(data_range[0], data_range[1], self.GridSize)
        self.UsedPoints = points
        self.solvers=cp.settings.SOLVERS

    def validate_non_decreasing(self, vector, sig_figs=4):
        is_non_decreasing = True
        for i in range(1, len(vector)):
            if round(vector[i], sig_figs) < round(vector[i-1], sig_figs):
                is_non_decreasing = False
        return is_non_decreasing

    def update_k_fold(self, k_fold=5, k_fold_repeats=5):
        self.k_fold = k_fold
        self.k_folds = get_k_folds(self.Variables, k_fold)
        self.num_repeats_k_fold = k_fold_repeats

    def update_symbolic_data(self, grid_size, data_range, co_variables, response, variable, df_quantile, k_fold_repeats, variables=[]):
        self.GridSize = grid_size
        self.DataRange = data_range
        self.co_variables = co_variables
        self.response = response
        self.variable_column = variable
        self.q = len(co_variables) + 1  # - 1  # + 1 to include beta0
        self.q_dsd = (2 * (len(co_variables) - 1)) + 1  # + 1 to include beta0
        self.Variables = variables
        if type(self.Variables) != list:
            self.Variables=self.Variables.tolist()
        self.N = len(self.Variables)
        self.quantile_df = df_quantile
        self.k_fold = 1
        self.k_folds = []
        self.is_k_fold = False
        self.num_repeats_k_fold = k_fold_repeats
        self.quantile_df['X_0'] = [
            [np.ones(self.GridSize), np.ones(self.GridSize)] for _ in range(self.N)]
        self.set_basis_info(self.n_bases, self.BasisType,
                            self.BasisDegreeOfSpline)
       
        num_test = int(self.N * .2)
        total = self.Variables
        self.test_variable = rd.sample(total, k=num_test)
        self.train_variables = remove_elements(total, self.test_variable)

    def set_train_test_variables(self, test):
        total = self.Variables
        self.test_variable = test
        self.train_variables = remove_elements(total, test)
        

    def get_mean_function(self, indexes):
        sum_value=np.zeros(self.GridSize)
        for j in indexes:
            sum_value+=self.quantile_df[self.response][j]

        return sum_value/len(indexes)

    def get_measured_error(self, y1, mean):
        return get_integral_square_function(y1-mean , self.UsedPoints)
    
    def set_basis_info(self, n_bases, basis_type, spline_of_degree):
        self.BasisType = basis_type
        self.BasisDegreeOfSpline = spline_of_degree
        self.n_bases = n_bases
        self.auxiliar_methods = Auxiliary.BasisFunctionsAuxiliary(
            self.GridSize, basis_type, self.UsedPoints, n_bases)
        self.get_basis_function()

    def get_basis_function(self):
        data_basis = {}
        data_range = self.DataRange
        basis_type = self.BasisType
        spline_of_degree = self.BasisDegreeOfSpline
        n_bases = self.n_bases
        if basis_type == "Fourier" or basis_type == "Fourier1":
            len_tau = data_range[1] - data_range[0]
            period = 2 * np.pi / len_tau
            self.auxiliar_methods.initiate_fourier(period, len_tau)
            f = int((n_bases - 1) / 2) + 1
            data_basis["phi_0"] = self.auxiliar_methods.approx_function(0, 0)
            j = 1
            for r in range(1, f):
                data_basis["phi_%s" %
                           j] = self.auxiliar_methods.approx_function(r, 0)
                j = j + 1
                data_basis["phi_%s" %
                           j] = self.auxiliar_methods.approx_function(r, 1)
                j += 1

            # df_basis = pd.DataFrame(data=data_basis)
        elif basis_type == "BSpline":
            zeros = np.zeros(spline_of_degree + 1)
            ones_vector = np.ones(spline_of_degree + 1)
            missing = (n_bases - 1) - spline_of_degree
            m = get_points(data_range[0] + .1, data_range[1] - .1, missing)
            spline_knots = np.concatenate((zeros, m, ones_vector))

            self.auxiliar_methods.initiate_b_splines(
                spline_of_degree, spline_knots)
            basis = self.auxiliar_methods.get_b_splines()
            for r in range(n_bases):
                data_basis["phi_%s" % r] = basis[r]

        else:
            for r in range(n_bases):
                data_basis["phi_%s" %
                           r] = self.auxiliar_methods.approx_function(r, 0)

        df_basis = pd.DataFrame(data=data_basis)
        self.Basis_df = df_basis
        self.n_bases = n_bases

    def get_fda_r2_with(self, variables, estimations):
        estimated_value = 0
        observed_value = 0
        var_index = get_indexes(self.Variables, variables)
        mean_response= get_mean_from_function(self.get_mean_function(var_index),self.UsedPoints)
        for i in var_index:
            estimated_value += self.get_measured_error(estimations["y_%s" % i], mean_response)
            observed_value += self.get_measured_error(self.quantile_df[self.response][i], mean_response)

        return (estimated_value / observed_value) if len(variables) > 0 else -1
    
    def get_fda_r2_functional_with(self,  variables, estimations):
        estimated_value = 0
        observed_value = 0
        var_index = get_indexes(self.Variables, variables)
        sacled_mean = self.get_mean_function(var_index)
        for i in var_index:
            estimated_value += self.get_measured_error(estimations["y_%s" % i], sacled_mean)
            observed_value += self.get_measured_error(self.quantile_df[self.response][i], sacled_mean)

        return (estimated_value / observed_value) if len(variables) > 0 else -1 
   
    def fit_fda(self, operator_type="N", lambda_value=0, n_bases=20, basis_type="BSpline", spline_of_degree=4):
        """
        Fit Concurrent FLM for our data

        Args:
            operator_type (str, optional): Type of penalization options ('D1','D2','D3','D4','D2D1','N'=none). Defaults to "N".
            lambda_value (int, optional): Range for Lambda. Defaults to 0.
            n_bases (int, optional): Number of basis functions to use. Defaults to 20.
            basis_type (str, optional): Basis function to use. Options ('BSpline','Fourier','Legendre','Pol','soft','Exponential'). Defaults to "BSpline".
            spline_of_degree (int, optional): Number of degrees for the Spline. Defaults to 4.
        """
        self.set_basis_info(n_bases, basis_type, spline_of_degree)

        mse_test, fda_r_2_test, non_decreasing, mse_train, r2_train,r2_train_f, r2_test_f = self.get_fda_un_penalized(
            self.train_variables, self.test_variable, operator_type=operator_type, lambda_value=lambda_value, is_principal=True, get_means_distance=True)
        self.unp_fda_train_mse = mse_train
        self.unp_fda_test_mse = mse_test
        self.fda_r_2_train = r2_train
        self.fda_r_2_test = fda_r_2_test
        self.NumberOfNonDecreasingResponses = non_decreasing
        self.fda_r2_train_f = r2_train_f
        self.fda_r_2_test_f = r2_test_f

    def get_fda_un_penalized(self, train_variables, test_variable, operator_type="N", lambda_value=0, is_principal=False, get_means_distance=False):
        variables = train_variables
        var_index = get_indexes(self.Variables, train_variables)
        test_index = get_indexes(self.Variables, test_variable)
        data_weights = {}
        my_block_r = []
        for i in range(self.q):
            for j in range(self.q):
                val = 0
                for t in var_index:
                    val += np.multiply(self.quantile_df["X_%s" %
                                       i][t][0], self.quantile_df["X_%s" % j][t][0])

                data_weights["w_%s%s" % (i, j)] = val

        for j in range(self.n_bases):
            for i in range(self.q):
                if operator_type != "N":
                    penal = self.get_linear_operator(
                        operator_type, self.Basis_df["phi_%s" % i])
                    my_block_r.append(
                        lambda_value * integrate.simpson(penal * penal, self.UsedPoints))
                else:
                    my_block_r.append(0)

        df_weights = pd.DataFrame(data=data_weights)
        data_xy = {}
        for i in range(self.q):
            val = 0
            for t in var_index:
                val += np.multiply(self.quantile_df["X_%s" % i]
                                   [t][0], self.quantile_df[self.response][t])

            data_xy["xy_%s" % i] = val

        df_xy = pd.DataFrame(data=data_xy)

        data_phis = {}
        for i in range(self.n_bases):
            for j in range(self.n_bases):
                data_phis["phi_%s_%s" % (i, j)] = np.multiply(
                    self.Basis_df["phi_%s" % i], self.Basis_df["phi_%s" % j])

        df_z = pd.DataFrame(data=data_phis)

        self.Weighted_df = df_weights
        self.Xy_df = df_xy
        self.Phi_by_Phi_df = df_z
        my_block = []
        my_xy_block = []
        for i in range(self.q):
            current = []
            my_xy_block.append(self.auxiliar_methods.calculate_mi_xy(
                i, df_xy, self.Basis_df).T)
            for j in range(self.q):
                current.append(self.auxiliar_methods.calculate_mi(
                    i, j, df_weights, df_z))
            my_block.append(current)

        px = np.block(my_block)
        py = np.block(my_xy_block)
        p_r = np.diag(my_block_r)

        x_inverse = np.linalg.inv(px + p_r)
        b = np.matmul(x_inverse, py)

        n_bases = self.n_bases
        data_estimation = {}
        for r in range(self.q):
            data_estimation["beta%s" % r] = self.auxiliar_methods.function_estimation(
                b[r * n_bases:(r + 1) * n_bases])

        df_beta = pd.DataFrame(data=data_estimation)

        y_h = {}
        for i in range(self.N):
            y_hat = np.zeros(self.GridSize)
            for r in range(self.q):
                y_hat += np.multiply(df_beta["beta%s" %
                                     r], self.quantile_df["X_%s" % r][i][0])
            y_h["y_%s" % i] = y_hat

        if len(test_variable)==1:
            ind=test_index[0]
            self.fda_loocv_estimations.append({'Penalty' : 'fda', 'Variable' : test_variable[0], 'Estimation' : y_h["y_%s" % ind]}, ignore_index = True)

        if is_principal:
            self.df_fda_y_hat = pd.DataFrame(y_h)
            self.Regression_Coefficients = b
            self.df_beta = df_beta
            
        #if get_means_distance:
        #    self.fda_d_means = get_error_function(self.mean_response,self.get_mean_estimated_function(y_h))

        mse_test = 0
        for i in test_index:
            mse_test = mse_test + \
                self.get_error(
                    self.quantile_df[self.response][i], y_h["y_%s" % i])

        fda_r_2_test = self.get_fda_r2_with(test_variable, y_h) 
        fda_r_2_test_f = self.get_fda_r2_functional_with(test_variable, y_h) 
        mse_train = 0
        for i in var_index:
            mse_train = mse_train + \
                self.get_error(
                    self.quantile_df[self.response][i], y_h["y_%s" % i])

        fda_r_2_train = self.get_fda_r2_with(variables, y_h) 
        fda_r_2_train_f = self.get_fda_r2_functional_with(variables, y_h) 

        non_decreasing = 0
        for i in range(self.N):
            if self.validate_non_decreasing(y_h["y_%s" % i], 4):
                non_decreasing += 1

        mse_t = mse_test / len(test_variable) if len(test_variable) > 0 else -1
        return mse_t, fda_r_2_test, non_decreasing, mse_train / len(variables), fda_r_2_train,fda_r_2_train_f, fda_r_2_test_f


    def get_matrix(self, train_variables):
        dim = (self.q - 1) * 2 + 1
        s = (dim, dim)
        matrix_q = np.zeros(s)
        matrix_q_sums = np.zeros(s)
        train_indexes = get_indexes(self.Variables, train_variables)
        for i in range(dim):
            if i == 0:
                current_i = 0
                i_index = 0
            elif i % 2 != 0:
                current_i += 1
                i_index = 0
            else:
                i_index = 1

            b_first_step = True
            for j in range(i, dim):
                val = np.zeros(self.GridSize)
                if b_first_step:
                    b_first_step = False
                    current_j = current_i
                elif j % 2 != 0:
                    current_j += 1

                for t in train_indexes:
                    j_index = 0 if (current_j == 0 or j % 2 != 0) else 1
                    val += np.multiply(self.quantile_df["X_%s" % current_i][t][i_index], self.quantile_df["X_%s" % current_j][t][j_index])
                    
                total = integrate.simpson(val, self.UsedPoints)
                matrix_q[i, j] = matrix_q[j, i] = total
                matrix_q_sums[i, j] = matrix_q_sums[i, j] = sum(val)

        r = np.zeros(dim)
        r_sums = np.zeros(dim)
        for k in range(dim):
            val = 0
            if k == 0:
                current_k = 0
            elif k % 2 != 0:
                current_k += 1

            for p in train_indexes:
                index_k = 0 if (k == 0 or k % 2 != 0) else 1
                val += np.multiply(self.quantile_df[self.response][p],
                                   self.quantile_df["X_%s" % current_k][p][index_k])

            r[k] = integrate.simpson(val, self.UsedPoints)
            r_sums[k] = sum(val)

        k_value = 0
        train_index=get_indexes(self.Variables,train_variables)
        for p in train_index:
            k_value += integrate.simpson(np.multiply(
                self.quantile_df[self.response][p], self.quantile_df[self.response][p]), self.UsedPoints)

        # Quadratic problem L(x)=-x.T*r+x.T*Q*x
        # x is the unknown vector of size n
        # r is a vector of the same size as x
        # Q is a square symmetric matrix of dimension n by n
        self.Q_Matrix = matrix_q
        self.r_Matrix = -1 * r.reshape((dim,))
        self.Q_sums_Matrix = matrix_q_sums
        self.r_sums_Matrix = r_sums
        self.K_value = k_value

    
    def fit_dsd(self, penalization="", max_l=3):
        """
        Fit DSD model to our Data. This can be with or without Penalizations

        Args:
            penalization (string, optional): Name of penalizations to apply. If empty then apply all. Options ('l1','l2','elastic','l1_comp','l2_comp','elastic_comp'). Defaults to "".
            max_l (int, optional): Range for Lambda factor on Penalizations. Defaults to 3.
        """ 
       
        betas = self.get_regression_un_penalized_coefficients(self.train_variables)
        self.unp_b = betas 
        estimations = self.get_dsd_y_hat(betas)
        self.unp_train_mse = self.get_mse_dsd(estimations, self.train_variables)
        self.unp_test_mse = self.get_mse_dsd(estimations, self.test_variable)
        self.unp_dsd_y_hat = estimations
        self.dsd_r2_train = self.get_dsd_r2_value(estimations, self.train_variables)
        self.dsd_r2_test = self.get_dsd_r2_value(estimations, self.test_variable)
        self.dsd_r2_train_f = self.get_dsd_r2_functional_value(estimations, self.train_variables)
        self.dsd_r2_test_f = self.get_dsd_r2_functional_value(estimations, self.test_variable)
        
    
    def get_regression_un_penalized_coefficients(self, train_variables): 
        self.get_matrix(train_variables)
        dim_a = (self.q - 1) * 2 + 1

        beta = cp.Variable(dim_a)
        u = [0]
        u = np.append(u, -1 * np.ones(2 * (self.q - 1)))
        G = np.diag(u)
        problem = cp.Problem(
            cp.Minimize((1 / 2) * cp.quad_form(beta,
                        self.Q_Matrix) + self.r_Matrix.T @ beta),
            [G @ beta <= 0])
        solver_index = 0
         
        while beta.value is None and solver_index<len(self.solvers):             
            try:
                problem.solve(solver=self.solvers[solver_index])
            except Exception :
                #print(error)
                pass #raise None
            finally:
                solver_index=solver_index+1
      
        return beta.value

    
    def get_estimated(self, beta, variable, round_factor=4):
        estimated = np.zeros(self.GridSize)
        parameter = 0
        current_b = 0
        dsd_q=(self.q-1)*2+1
        while current_b < dsd_q:
            if round(beta[current_b], round_factor) > 0:
                    estimated = estimated + np.multiply(self.quantile_df["X_%s" %parameter][variable][0],beta[current_b])
            if current_b != 0:
                current_b = current_b + 1
                if round(beta[current_b], round_factor) > 0:
                    estimated = estimated + np.multiply(self.quantile_df["X_%s" %parameter][variable][1],beta[current_b])
            
            current_b = current_b + 1
            parameter = parameter + 1
        return estimated
 

    def get_dsd_y_hat(self,betas):
        y_h = {}
        for i in range(len(self.Variables)):
            y_h["y_%s" % i] = self.get_estimated(betas, i)

        return pd.DataFrame(data=y_h)
    
    def get_error(self, y1, y2):
        return get_function_error(y1 ,y2, self.UsedPoints)
    
    
    def get_mse_dsd(self, estimations, variables):
        mse = 0
        var_indexes = get_indexes(self.Variables, variables)
        for variable in var_indexes:
            response = self.quantile_df[self.response][variable]
            estimated_response = estimations["y_%s"%variable]
            mse = mse + self.get_error(response, estimated_response)
        return mse / len(variables)

    def get_dsd_r2_value(self, estimations , variables):
        estimated_value = 0
        observed_value = 0
        indexes = get_indexes(self.Variables,variables) 
        sacled_mean = get_mean_from_function(self.get_mean_function(indexes),self.UsedPoints)# self.mean_response_value#(self.mean_response_value + self.get_estimated_dsd_y_mean(betas))/2
        for i in indexes:
            estimated=estimations["y_%s"%i]
            observed = self.quantile_df[self.response][i]
            estimated_value += self.get_measured_error(estimated, sacled_mean)# self.mean_response_value)
            observed_value += self.get_measured_error(observed, sacled_mean)# self.mean_response_value)
        
        return (estimated_value / observed_value)
    
    def get_dsd_r2_functional_value(self, estimations, variables):
        estimated_value = 0
        observed_value = 0
        indexes = get_indexes(self.Variables,variables) 
        sacled_mean = self.get_mean_function(indexes) 
        for i in indexes:
            estimated=estimations["y_%s"%i]
            observed = self.quantile_df[self.response][i]
            estimated_value += self.get_measured_error(estimated,sacled_mean)# self.mean_response_value)
            observed_value += self.get_measured_error(observed, sacled_mean)# self.mean_response_value)
        
        return (estimated_value / observed_value)
 

