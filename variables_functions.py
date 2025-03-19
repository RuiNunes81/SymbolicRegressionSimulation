import warnings
warnings.filterwarnings('ignore')
import KernelDensities as Kde
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from random import sample
import random as rd
import SymbolicDensityRegression as sdr 
pd.set_option('display.max_colwidth', None)


### Some auxiliar functions ###
def get_points(lower, upper, n):
    """General function to get n points between lower and upper

    Args:
        lower (number): lower limit
        upper (number): upper limit
        n (int): number of points to generate

    Returns:
        list: list of generated values
    """
    delta = (upper - lower) / n
    p = []
    for i in range(n):
        p.append(lower + i * delta)

    return np.array(p)

simulation_columns = ['Simulation',
                      'n_individuals',
                      'Variance factor',
                      'Distributions',
                      'Error',
                      'Test Indiv.',
                      'Coefficient Config',                      
                      'FDA Train MSE',
                      'FDA Test MSE',
                      'FDA R2(training)',
                      'FDA R2(test)',
                      'FDA R2_functional(training)',
                      'FDA R2_functional(test)',
                      'FDA Non-Decreasing Responses',
                      'DSD Train MSE',
                      'DSD Test MSE',
                      'DSD R2 Training',
                      'DSD R2 Test',
                      'DSD R2_functional Training',
                      'DSD R2_functional Test',
                      'DSD Coefs',
                      'lst DSD Coefs',
                      "Estimations"
                    ]

def get_simulation_response(sufix,test_ind,data_range, GridSize ,simulation, data_Qf, co_variables, kfoldRepeats,
                           n_individuals, n_regressors, num_data_points, CofficientsConfig,variables, response, variable,error,gen_type_des,
                            var_factor,sdd):
   
    try:
        regression = sdr.SymbolicDensityRegression(GridSize,data_range)
        
        simulation_detail_data=[]
        
        regression.update_symbolic_data(GridSize, data_range, co_variables, response, variable , data_Qf, 1,variables)
        regression.set_train_test_variables(test_ind)
        
        try:
            regression.fit_fda()
        except Exception as exception:
            input(exception)
            
        regression.fit_dsd()
        tr_mse=regression.unp_train_mse
        tst_mse=regression.unp_test_mse
        tr_r2=regression.dsd_r2_train
        tst_r2=regression.dsd_r2_test
        tr_r2_f=regression.dsd_r2_train_f
        tst_r2_f=regression.dsd_r2_test_f
        tr_fda_mse=regression.unp_fda_train_mse
        tst_fda_mse=regression.unp_fda_test_mse
        tr_fda_r2=regression.fda_r_2_train
        tst_fda_r2=regression.fda_r_2_test
        tr_fda_r2_f=regression.fda_r2_train_f
        tst_fda_r2_f=regression.fda_r_2_test_f
        ndf_fda=regression.NumberOfNonDecreasingResponses
        dsd_coef=regression.unp_b
        
        str_coef = "$v=%s"%round(dsd_coef[0],3) 
        for i in range(int((len(dsd_coef)-1)/2)):
            str_coef = str_coef + ", \; a_%s=%s, \; b_%s=%s"%(i,round(dsd_coef[2*i+1],3),i,round(dsd_coef[2*i+2],3)) 
        str_coef=str_coef+"$"
        full=str(dsd_coef)

        simulation_detail_data.append(
            ("%s" % simulation,
                 n_individuals,
                var_factor,
                "%s" % gen_type_des,
                error,
                test_ind,
                CofficientsConfig,
                tr_fda_mse,
                tst_fda_mse,
                #regression.unp_fda_mse,
                tr_fda_r2,
                tst_fda_r2,
                tr_fda_r2_f,
                tst_fda_r2_f,
                ndf_fda,
                tr_mse,
                tst_mse,
                tr_r2,
                tst_r2,
                tr_r2_f,
                tst_r2_f,
                str_coef,
                 full,
                regression.unp_dsd_y_hat
                ))
        sdd.append(
            ("%s" % simulation,
                n_individuals,
                var_factor,
                "%s" % gen_type_des,
                error,
                test_ind,
                CofficientsConfig,
                tr_fda_mse,
                tst_fda_mse,
                #regression.unp_fda_mse,
                tr_fda_r2,
                tst_fda_r2,
                tr_fda_r2_f,
                tst_fda_r2_f,
                ndf_fda,
                tr_mse,
                tst_mse,
                tr_r2,
                tst_r2,
                tr_r2_f,
                tst_r2_f,
                str_coef,
                full,
                regression.unp_dsd_y_hat
                ))
    
        df_simulation_detail = pd.DataFrame(
            simulation_detail_data, columns=simulation_columns)
        with open("Simulation_%s_%s.csv" % ("Detail",sufix), 'a') as f:
            df_simulation_detail.to_csv(
                f, header=f.tell() == 0, line_terminator='\n')
    except Exception as exception:
        input(exception)
        with open("errors_means_%s.txt" % (sufix), 'a') as f:
            f.write("====================================\n")
            f.write(exception)
            f.write("\n")
            f.write("====================================\n")


def get_config_str(confg):
    i=1
    lst=confg
    b=lst[0] 
    par=lst[1] 
    str = "$v=%s"%b
    for i in range(len(par)):
        str = str + ",\; a_%s=%s, \; b_%s=%s"%((i+1),par[i][0],(i+1),par[i][1])
    str=str+"$"
    return(str)






### Some variables ###
dist_coef = [0.4,1.4] 
num_reg=[1,1,1,3,3]
GridSize=500
num_data_points=5000
data_range=[0,1]
n_individuals=50 
kernel ="gaussian"
#Centrality for variables
mean_var = [20,10,5]
vari_var = [8,6,4]

coefficients = [[-20, [[7, 2]]],
                [141,  [[1, 4]]],
                [29,  [[6, 7]]],
                [-15, [[5, 1], [8, 3], [1, 5]]],
                [2,  [[3, 1], [2, 3], [1, .5]]]]

points = get_points(data_range[0], data_range[1], GridSize)
dummy_response=np.zeros(GridSize)
dummy_ones=np.ones(GridSize)
factors=[0.10, .25, .5]#[0.05,0.10,0.25, 0.5, 0.95] 
var_factors=[0.1,0.15,0.25]#[0.1,0.15,0.25,0.25,0.5,1]

responses_variables = ["Y"]


cols = ['Individual','Distribution']
  
for col in range(3):
    cols.append("X_data_%s" % (col+1))    

cols.append("Means")

for col in range(len(coefficients)):
    cols.append("C_%s" % (col+1))

for col in range(len(coefficients)):
    for mu in range(len(factors)):
        for vu in range(len(var_factors)):
            cols.append("M_%s%s%s" % (col+1,mu,vu))
            cols.append("V_%s%s%s" % (col+1,mu,vu))
        

index = 0
data_Qf_unif = pd.DataFrame(columns=cols) 
data_Qf_normal = pd.DataFrame(columns=cols)
data_Qf_log = pd.DataFrame(columns=cols)

indi=["ind%s"%i for i in range(n_individuals)]

data_Qf_unif["Individual"],data_Qf_normal["Individual"],data_Qf_log["Individual"]=indi,indi,indi 

data_Qf_unif["Distribution"]=["Uniforms"]*n_individuals
data_Qf_normal["Distribution"]=["Normals"]*n_individuals
data_Qf_log["Distribution"]=["LogNormals"]*n_individuals

cols_df = ['Individual','Distribution']  

for col in range(3):
    cols_df.append("X_%s" % (col+1)) 
    
for col in range(len(coefficients)):
    cols_df.append("Y_%s" % (col+1))

for col in range(len(coefficients)):
    for mu in range(len(factors)):
        for vu in range(len(var_factors)):
            cols_df.append("Y_u_%s%s%s" % (col+1,mu,vu))
            cols_df.append("Y_n_%s%s%s" % (col+1,mu,vu))
            cols_df.append("Y_l_%s%s%s" % (col+1,mu,vu))
        

index = 0
data_unif_Qf = pd.DataFrame(columns=cols_df) 
data_normal_Qf = pd.DataFrame(columns=cols_df)
data_log_Qf = pd.DataFrame(columns=cols_df)

data_unif_Qf["Individual"],data_normal_Qf["Individual"],data_log_Qf["Individual"]=indi,indi,indi 
data_unif_Qf["Distribution"],data_normal_Qf["Distribution"],data_log_Qf["Distribution"]=["Uniforms"]*n_individuals,["Normals"]*n_individuals,["LogNormals"]*n_individuals

            
#### Some data generation ####
uniforms,normals,lognormals=[],[],[] 
means_uniforms,means_normals,means_lognormals=[],[],[]
var_uniforms,var_normals,var_lognormals=[],[],[] 

u_1,u_2,u_3,u_4,u_5=[],[],[],[],[]
n_1,n_2,n_3,n_4,n_5=[],[],[],[],[]
l_1,l_2,l_3,l_4,l_5=[],[],[],[],[]


kde = Kde.KernelDensities(GridSize) 

means_=[[np.random.uniform(dist_coef[0]*mean_var[l],dist_coef[1]*mean_var[l]) for l in range(3)] for _ in range(n_individuals)]
vari_=[[np.random.uniform(dist_coef[0]*vari_var[l],dist_coef[1]*vari_var[l]) for l in range(3)] for _ in range(n_individuals)]

err_means=[]
for ind in range(n_individuals):
    cn=0
    cr_mean=[]
    for coef in coefficients:
        mn=coef[0]
        cf=coef[1]
        index=0 
        for cf in coef[1]:  
            mn=mn+(cf[0]-cf[1])*means_[ind][index] 
            index=index+1
        
        cr_mean.append(mn) 
    err_means.append(cr_mean)


uniforms=[[np.random.uniform(means_[ind][k]-np.sqrt(3*vari_[ind][k]),means_[ind][k]+np.sqrt(3*vari_[ind][k]),num_data_points) for k in range(3)] for ind in range(n_individuals)]
normals=[[np.random.normal(means_[ind][k],np.sqrt(vari_[ind][k]),num_data_points) for k in range(3)] for ind in range(n_individuals)]
lognormals=[[np.random.lognormal(0.5*np.log(means_[ind][k]**4/(vari_[ind][k]+means_[ind][k]**2)), np.sqrt(np.log(1+(vari_[ind][k]/means_[ind][k]**2))),num_data_points) for k in range(3)] for ind in range(n_individuals)]
 
err_uniforms=[]
err_normals=[]
err_lognormals=[]
for ind in range(n_individuals):
    curr_u,curr_n,curr_l=[],[],[]
    for k in range(len(coefficients)):
        cf_u,cf_n,cf_l=[],[],[]
        for mu in factors:
            m_u,m_n,m_l=[],[],[]
            for vr in var_factors: 
                m_u.append(np.random.uniform(mu*err_means[ind][k]-np.sqrt(3*vr*mu*err_means[ind][k]),mu*err_means[ind][k]+np.sqrt(3*vr*mu*err_means[ind][k]),num_data_points))
                m_n.append(np.random.normal(mu*err_means[ind][k],np.sqrt(vr*mu*err_means[ind][k]),num_data_points))
                m_l.append(np.random.lognormal(0.5*np.log((mu*err_means[ind][k])**4/((vr*mu*err_means[ind][k])+(mu*err_means[ind][k])**2)), np.sqrt(np.log(1+((vr*mu*err_means[ind][k])/(mu*err_means[ind][k])**2))),num_data_points))
            cf_u.append(m_u)
            cf_n.append(m_n)
            cf_l.append(m_l)
        curr_u.append(cf_u)
        curr_n.append(cf_n)
        curr_l.append(cf_l)
    err_uniforms.append(curr_u)
    err_normals.append(curr_n)
    err_lognormals.append(curr_l)