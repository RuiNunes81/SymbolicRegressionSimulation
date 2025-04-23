import random as rd 
import cvxpy as cp
import matplotlib.pyplot as plt
import pandas as pd
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
    # Start - General functions
    ##############################
    def get_error(self, y1, y2):
        return get_function_error(y1 ,y2, self.UsedPoints)
     
    def validate_non_decreasing(self, vector, sig_figs=4):
        is_non_decreasing = True
        for i in range(1, len(vector)):
            if round(vector[i], sig_figs) < round(vector[i-1], sig_figs):
                is_non_decreasing = False
        return is_non_decreasing

    def get_non_decreasing_function(self):
        non_decreasing_functions = 0
        for i in range(self.N):
            if self.validate_non_decreasing(self.df_fda_y_hat[f"y_{i}"]):
                non_decreasing_functions += 1
        return non_decreasing_functions

    ##############################
    # End - General functions
    ##############################

    ##############################
    # Start - Initialize Functions
    ##############################
    def InitializeVariables(self, grid_size, data_range):
        self.GridSize = grid_size
        self.DataRange = data_range
        points = get_points(data_range[0], data_range[1], self.GridSize)
        self.UsedPoints = points
        
        self.num_repeats_k_fold = 5

    def __init__(self, grid_size, data_range):
        self.InitializeVariables(grid_size, data_range)        
        self.solvers=cp.settings.SOLVERS
        
        
    def update_symbolic_data(self, grid_size, data_range, co_variables, response, variable, df_quantile, variables=[]):
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

        df_quantile['X_0'] = [[np.ones(grid_size), np.ones(grid_size)] for _ in range(len(variables))]
        
        self.quantile_df = df_quantile
        num_test = int(len(self.Variables) * .2)
        total = self.Variables
        self.test_variable = rd.sample(total, k=num_test)
        self.train_variables = remove_elements(total, self.test_variable)

    def set_train_test_variables(self, test):
        total = self.Variables
        self.test_variable = test
        self.train_variables = remove_elements(total, test)

    def get_mean_function(self, indexes):
        sum_value = np.sum([self.quantile_df[self.response][j] for j in indexes], axis=0)
        return sum_value / len(indexes)
         
    
    def get_measured_error(self, f_1, f_2):
        return integrate.simpson( (f_1-f_2)**2 , x=self.UsedPoints)
    

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
                    val += np.multiply(self.quantile_df[f"X_{current_i}"][t][i_index], self.quantile_df[f"X_{current_j}"][t][j_index])
                    
                total = integrate.simpson(val, x=self.UsedPoints)
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
                                   self.quantile_df[f"X_{current_k}"][p][index_k])

            r[k] = integrate.simpson(val, x=self.UsedPoints)
            r_sums[k] = sum(val)

        k_value = 0
        train_index=get_indexes(self.Variables,train_variables)
        for p in train_index:
            k_value += integrate.simpson(np.multiply(
                self.quantile_df[self.response][p], self.quantile_df[self.response][p]), x=self.UsedPoints)

        # Quadratic problem L(x)=-x.T*r+x.T*Q*x
        # x is the unknown vector of size n
        # r is a vector of the same size as x
        # Q is a square symmetric matrix of dimension n by n
        self.Q_Matrix = matrix_q
        self.r_Matrix = -1 * r.reshape((dim,))
        self.Q_sums_Matrix = matrix_q_sums
        self.r_sums_Matrix = r_sums
        self.K_value = k_value
     

    def fit_dsd(self):
        """
        Fit DSD model to our Data. This can be with or without Penalizations

        Args:
            penalization (string, optional): Name of penalizations to apply. If empty then apply all. Options ('l1','l2','elastic','l1_comp','l2_comp','elastic_comp'). Defaults to "".
            max_l (int, optional): Range for Lambda factor on Penalizations. Defaults to 3.
        """ 
       
        betas = self.get_regression_un_penalized_coefficients(self.train_variables)
        self.unp_b = betas 
        estimations = self.get_dsd_y_hat(betas)
        self.unp_train_mse = self.get_mse_dsd(betas, self.train_variables)
        self.unp_test_mse = self.get_mse_dsd(betas, self.test_variable)
        self.unp_dsd_y_hat = estimations
        self.dsd_r2_train = self.get_dsd_r2_value(betas, estimations, self.train_variables)
        self.dsd_r2_test = self.get_dsd_r2_value(betas, estimations, self.test_variable)
        
        self.dsd_train_TSS,self.dsd_train_RSS = self.get_dsd_TSS_RSS(betas, estimations, self.train_variables)
        self.dsd_test_TSS,self.dsd_test_RSS = self.get_dsd_TSS_RSS(betas, estimations, self.test_variable)
        #self.dsd_r2_train_f = self.get_dsd_r2_value_functional(betas, self.train_variables)
        #self.dsd_r2_test_f = self.get_dsd_r2_value_functional(betas, self.test_variable)
        # self.dsd_r2_train_v2,self.sum_obs_est_trt_v2,self.sum_obs_mean_trt_v2,self.sum_est_mean_trt_v2 = self.get_dsd_r2_value_v2(betas, self.train_variables)
        # self.dsd_r2_test_v2,self.sum_obs_est_tst_v2,self.sum_obs_mean_tst_v2,self.sum_est_mean_tst_v2  = self.get_dsd_r2_value_v2(betas, self.test_variable)
        self.dsd_r2_train_v2 = self.get_dsd_r2_value_v2(betas, estimations, self.train_variables)
        self.dsd_r2_test_v2 = self.get_dsd_r2_value_v2(betas, estimations, self.test_variable)
        self.dsd_r2_adjust_train = self.get_dsd_adjusted_r2_value(betas, estimations, self.train_variables)
        self.dsd_r2_oos_v2 = self.get_dsd_r2_oos_v2(betas)
        #self.dsd_r2_train_v3 = self.get_dsd_r2_value_v3(betas, self.train_variables)
        
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
                pass #raise None
            finally:
                solver_index=solver_index+1
      
        return beta.value
    
    def get_estimated(self, betas, variable, round_factor=4):
        estimated = np.zeros(self.GridSize)
        current_b=0
        while current_b < self.q:
            if current_b==0: 
                bt= betas[current_b] if round(betas[current_b],round_factor)>0 else 0
                estimated = estimated + np.multiply(self.quantile_df[f"X_{current_b}"][variable][0],bt)
            else: 
                bt1= betas[2*(current_b-1)+1] if round(betas[2*(current_b-1)+1],round_factor)>0 else 0
                bt2= betas[2*(current_b-1)+2] if round(betas[2*(current_b-1)+2],round_factor)>0 else 0
                estimated = estimated + np.multiply(self.quantile_df[f"X_{current_b}"][variable][0],bt1)+ np.multiply(self.quantile_df[f"X_{current_b}"][variable][1],bt2)   
            current_b = current_b + 1 

        return estimated
    
    def get_dsd_y_hat(self,betas):
        y_h = {}
        for i in range(len(self.Variables)):
            y_h[f"y_{i}"] = self.get_estimated(betas, i)

        return pd.DataFrame(data=y_h)
    
    def get_mse_dsd(self, betas, variables):
        mse = 0
        var_indexes = get_indexes(self.Variables, variables)
        for variable in var_indexes:
            observed = self.quantile_df[self.response][variable]
            estimated_response = self.get_estimated(betas,variable)
            mse = mse + integrate.simpson((observed-estimated_response)** 2,x=self.UsedPoints)
        return mse / len(variables)

    def get_dsd_r2_value(self, betas , estimations, variables):
        estimated_value = 0
        observed_value = 0
        indexes = get_indexes(self.Variables,variables) 
        sacled_mean = self.get_mean_function(indexes)
        for i in indexes:
            #estimated=self.get_estimated(betas,i)
            estimated=estimations[f"y_{i}"]
            observed = self.quantile_df[self.response][i]
            estimated_value += integrate.simpson((estimated-sacled_mean)** 2,x=self.UsedPoints)
            observed_value += integrate.simpson((observed-sacled_mean)** 2,x=self.UsedPoints)
        
        return (estimated_value / observed_value)
    
    def get_dsd_TSS_RSS(self, betas , estimations, variables):
        tss = 0
        rss = 0
        indexes = get_indexes(self.Variables,variables) 
        sacled_mean = self.get_mean_function(indexes)
        for i in indexes:
            #estimated=self.get_estimated(betas,i)
            estimated=estimations[f"y_{i}"]
            observed = self.quantile_df[self.response][i]
            rss += integrate.simpson((observed-estimated)** 2,x=self.UsedPoints)
            tss += integrate.simpson((observed-sacled_mean)** 2,x=self.UsedPoints)
        
        return tss, rss
    # def get_dsd_r2_value_functional(self, betas , variables):
    #     estimated_value = np.zeros(self.GridSize)
    #     observed_value = np.zeros(self.GridSize)
    #     indexes = get_indexes(self.Variables,variables) 
    #     sacled_mean = np.mean(self.quantile_df[self.response][indexes], axis=0)# self.get_mean_function(indexes)
    #     for i in indexes:
    #         estimated=self.get_estimated(betas,i)
    #         observed = self.quantile_df[self.response][i]
    #         estimated_value = estimated_value + (observed-estimated)** 2 # self.get_measured_error(estimated, sacled_mean)# self.mean_response_value)
    #         observed_value = observed_value + (observed-sacled_mean)** 2 # self.get_measured_error(observed, sacled_mean)# self.mean_response_value)
        
    #     return integrate.simpson(1-(estimated_value/observed_value),x=self.UsedPoints) #(estimated_value / observed_value)

    def get_dsd_r2_value_v2(self, betas , estimations, variables):
        estimated_value = 0
        estimated_2_mean_value = 0
        observed_value = 0
        indexes = get_indexes(self.Variables,variables) 
        sacled_mean = self.get_mean_function(indexes)
        for i in indexes:
            #estimated=self.get_estimated(betas,i)
            estimated=estimations[f"y_{i}"]
            observed = self.quantile_df[self.response][i]
            estimated_value += integrate.simpson((observed-estimated)** 2,x=self.UsedPoints)# self.get_measured_error(estimated, sacled_mean)# self.mean_response_value)
            observed_value += integrate.simpson((observed-sacled_mean)** 2,x=self.UsedPoints)# self.get_measured_error(observed, sacled_mean)# self.mean_response_value)
            #estimated_2_mean_value += integrate.simpson((estimated-sacled_mean)** 2,x=self.UsedPoints)
        
        return 1-(estimated_value / observed_value)#,estimated_value,observed_value,estimated_2_mean_value]
     
    
    def get_dsd_adjusted_r2_value(self, betas, estimations , variables):  
        r2=self.get_dsd_r2_value_v2( betas ,estimations, variables)
        return 1- (((len(variables)-1)*(1-r2))/(len(variables)-1-self.q))
        
    def get_dsd_r2_oos_v2(self, betas):
        estimated_value = 0
        observed_value = 0
        indexes = get_indexes(self.Variables,self.test_variable) 
        indexes_train = get_indexes(self.Variables,self.train_variables) 
        sacled_mean = self.get_mean_function(indexes_train)
        for i in indexes:
            estimated=self.get_estimated(betas,i)
            observed = self.quantile_df[self.response][i]
            estimated_value += integrate.simpson((estimated-observed)** 2,x=self.UsedPoints)# self.get_measured_error(estimated, sacled_mean)# self.mean_response_value)
            observed_value += integrate.simpson((sacled_mean-observed)** 2,x=self.UsedPoints)# self.get_measured_error(observed, sacled_mean)# self.mean_response_value)
        
        return 1-(estimated_value / observed_value)

    ##### FDA ##########
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
                data_basis[f"phi_{j}"] = self.auxiliar_methods.approx_function(r, 0)
                j = j + 1
                data_basis[f"phi_{j}"] = self.auxiliar_methods.approx_function(r, 1)
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
                data_basis[f"phi_{r}"] = basis[r]

        else:
            for r in range(n_bases):
                data_basis["phi_{r}"] = self.auxiliar_methods.approx_function(r, 0)

        df_basis = pd.DataFrame(data=data_basis)
        self.Basis_df = df_basis
        self.n_bases = n_bases

    def plot_fda_basis_function(self, plot_n_basis):
        points = self.UsedPoints

        for r in range(plot_n_basis):
            plt.plot(points, self.Basis_df[f"phi_{r}"], label="$\\phi_{%s}(x)$" % r)

        plt.title("Basis Function $\\phi_{k}(x)$")
        if plot_n_basis <= 6:
            plt.legend()

    def get_fda_r2_v2_with(self, variables, estimations):
        estimated_value = 0
        observed_value = 0
        var_index = get_indexes(self.Variables, variables)
        sacled_mean = self.get_mean_function(var_index)
        for i in var_index:
            observed = self.quantile_df[self.response][i]
            estimated_value += integrate.simpson((observed-estimations[f"y_{i}"])** 2,x=self.UsedPoints)# self.get_measured_error(estimated, sacled_mean)# self.mean_response_value)
            observed_value += integrate.simpson((observed-sacled_mean)** 2,x=self.UsedPoints)# self.get_measured_error(observed, sacled_mean)# self.mean_response_value)
        

        return 1-(estimated_value / observed_value) if len(variables) > 0 else -1
    
    def get_fda_r2_with(self,  variables, estimations):
        estimated_value = 0
        observed_value = 0
        var_index = get_indexes(self.Variables, variables)
        sacled_mean = self.get_mean_function(var_index)
        for i in var_index:
            estimated=estimations[f"y_{i}"]
            observed = self.quantile_df[self.response][i]
            estimated_value += integrate.simpson((estimated-sacled_mean)** 2,x=self.UsedPoints)# self.get_measured_error(estimated, sacled_mean)# self.mean_response_value)
            observed_value += integrate.simpson((observed-sacled_mean)** 2,x=self.UsedPoints)# self.get_measured_error(observed, sacled_mean)# self.mean_response_value)
        
        return (estimated_value / observed_value)   
    
    def get_fda_r2_loocv(self):
        val = 0
        val_mean = 0
        estimations=self.fda_loocv_estimations
        estimation_index = estimations.index
        variables = estimations.Variable
        estimates = estimations.Estimation
        for i in estimation_index: 
            index = get_indexes(self.Variables,[variables[i]])[0] 
            estimated=estimates[i]             
            resp = self.quantile_df[self.response][index] 
            val += self.get_measured_error(estimated, self.mean_response_value)
            val_mean += self.get_measured_error(resp, self.mean_response_value)

        return (val / val_mean) if len(variables) > 0 else -1

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
                #print("FDA Train-Test")
        mse_test, fda_r_2_test, non_decreasing, mse_train, r2_train,r2_train_v2, r2_test_v2 = self.get_fda_un_penalized(
            self.train_variables, self.test_variable, operator_type=operator_type, lambda_value=lambda_value, is_principal=True, get_means_distance=True)
        self.unp_fda_train_mse = mse_train
        self.unp_fda_test_mse = mse_test
        self.fda_r_2_train = r2_train
        self.fda_r_2_test = fda_r_2_test
        self.NumberOfNonDecreasingResponses = non_decreasing
        self.fda_r2_train_v2 = r2_train_v2
        self.fda_r_2_test_v2 = r2_test_v2

     

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
                    val += np.multiply(self.quantile_df[f"X_{i}"][t][0], self.quantile_df[f"X_{j}"][t][0])

                data_weights["w_%s%s" % (i, j)] = val

        for j in range(self.n_bases):
            for i in range(self.q):
                if operator_type != "N":
                    penal = self.get_linear_operator(
                        operator_type, self.Basis_df[f"phi_{i}"])
                    my_block_r.append(
                        lambda_value * integrate.simpson(penal * penal, x=self.UsedPoints))
                else:
                    my_block_r.append(0)
        
        #data_weights = np.atleast_1d(data_weights)
        #df_weights = pd.DataFrame(data=data_weights)
        df_weights = pd.DataFrame(data=data_weights, index=[0] if np.isscalar(data_weights) else None)
        data_xy = {}
        for i in range(self.q):
            val = 0
            for t in var_index:
                val += np.multiply(self.quantile_df[f"X_{i}"][t][0], self.quantile_df[self.response][t])

            data_xy[f"xy_{i}"] = val
        #data_xy = np.atleast_1d(data_xy)
        df_xy = pd.DataFrame(data=data_xy)

        data_phis = {}
        for i in range(self.n_bases):
            for j in range(self.n_bases):
                data_phis[f"phi_{i}_{j}"] = np.multiply(self.Basis_df[f"phi_{i}"], self.Basis_df[f"phi_{j}"])

        df_z = pd.DataFrame(data=data_phis)

        self.Weighted_df = df_weights
        self.Xy_df = df_xy
        self.Phi_by_Phi_df = df_z
        my_block = []
        my_xy_block = []
        for i in range(self.q):
            current = []
            my_xy_block.append(self.auxiliar_methods.calculate_mi_xy(i, df_xy, self.Basis_df).T)
            for j in range(self.q):
                current.append(self.auxiliar_methods.calculate_mi(i, j, df_weights, df_z))
            my_block.append(current)

        px = np.block(my_block)
        py = np.block(my_xy_block)
        p_r = np.diag(my_block_r)

        x_inverse = np.linalg.inv(px + p_r)
        b = np.matmul(x_inverse, py)

        n_bases = self.n_bases
        data_estimation = {}
        for r in range(self.q):
            data_estimation[f"beta{r}"] = self.auxiliar_methods.function_estimation(
                b[r * n_bases:(r + 1) * n_bases])

        df_beta = pd.DataFrame(data=data_estimation)

        y_h = {}
        for i in range(self.N):
            y_hat = np.zeros(self.GridSize)
            for r in range(self.q):
                y_hat += np.multiply(df_beta[f"beta{r}"], self.quantile_df[f"X_{r}"][i][0])
            y_h[f"y_{i}"] = y_hat
 
        if is_principal:
            self.df_fda_y_hat = pd.DataFrame(y_h)
            self.Regression_Coefficients = b
            self.df_beta = df_beta
         
        mse_test = 0
        for i in test_index:
            mse_test = mse_test + \
                self.get_error(
                    self.quantile_df[self.response][i], y_h[f"y_{i}"])

        fda_r_2_test = self.get_fda_r2_with(test_variable, y_h) 
        fda_r_2_test_v2 = self.get_fda_r2_v2_with(test_variable, y_h) 
        mse_train = 0
        for i in var_index:
            mse_train = mse_train + \
                self.get_error(
                    self.quantile_df[self.response][i], y_h[f"y_{i}"])

        fda_r_2_train = self.get_fda_r2_with(variables, y_h) 
        fda_r_2_train_v2 = self.get_fda_r2_v2_with(variables, y_h) 

        non_decreasing = 0
        for i in range(self.N):
            if self.validate_non_decreasing(y_h[f"y_{i}"], 4):
                non_decreasing += 1

        mse_t = mse_test / len(test_variable) if len(test_variable) > 0 else -1
        return mse_t, fda_r_2_test, non_decreasing, mse_train / len(variables), fda_r_2_train,fda_r_2_train_v2, fda_r_2_test_v2

    # Plots
    def plot_fda_estimated_coefficient(self):
        """
        data_basis = {}
        n_bases = self.n_bases
        covar=int(((self.q-1)/2))+1
        for r in range(covar):
            data_basis["beta%s" % r] = \
                self.auxiliar_methods.function_estimation(
                    self.Regression_Coefficients[r * n_bases:(r + 1) * n_bases])

        df_beta = pd.DataFrame(data=data_basis)
        """
        df_beta =self.df_beta 
        for r in range(self.q):
            plt.plot(self.UsedPoints,
                     df_beta[f"beta{r}"], label="$\\hat{\\beta}_{%s}(x)$" % r)

        plt.title("Coefficients")
        plt.legend()
        plt.show()

    def plot_fda_estimation_for_individual(self, individuals=[], plot_observed=False, plot_mean=False, save_plot=False):
        var_index = get_indexes(self.Variables, individuals)
        sacled_mean = self.get_mean_function(var_index)
        for r in var_index:
            plt.plot(
                self.UsedPoints, self.df_fda_y_hat["y_%s" % r], label="$\\Psi_{hat{y}_{%s}}(x)$" % r)
            if plot_observed:
                plt.plot(
                    self.UsedPoints, self.quantile_df[self.response][r], label="$\\Psi_{y_{%s}}(x)$" % r)
            if plot_observed:
                plt.plot(self.UsedPoints, sacled_mean,
                         label="$\\bar_{y}(x)$")
            plt.title("%s - Estimation" % self.Variables[r])
            plt.legend()
            if save_plot:
                plt.savefig("plot_fda_estimation_%s.png" % self.Variables[r])
            plt.show()

    ##############################
    # End - FDA
    ##############################
        
