import logging
import warnings
import numpy as np
import pandas as pd
import random as rd
import KernelDensities as Kde
import SymbolicDensityRegression as sdr
from multiprocessing import Process

warnings.filterwarnings('ignore')
pd.set_option('display.max_colwidth', None)

def compute_quantiles(args):
    kde, distribution = args
    return [kde.get_quantiles(distribution[ind, k]) for k in range(3)]

def generate_synthetic_data(n_individuals, num_data_points):
    means_ = np.random.uniform(dist_coef[0] * mean_var, dist_coef[1] * mean_var, (n_individuals, 3))
    stds = np.random.uniform(dist_coef[0] * sigmas, dist_coef[1] * sigmas, (n_individuals, 3))
    val_means = means_
    means_ = means_[:, :, np.newaxis]
    stds = stds[:, :, np.newaxis]
    uniforms = np.random.uniform(means_ - np.sqrt(3) * stds, means_ + np.sqrt(3) * stds, (n_individuals, 3, num_data_points))
    normals = np.random.normal(means_, stds, (n_individuals, 3, num_data_points))
    lognormals = np.random.lognormal(
        0.5 * np.log(means_**4 / np.sqrt(stds**2 + means_**2)),
        np.sqrt(np.log(1 + (stds**2 / means_**2))),
        (n_individuals, 3, num_data_points)
    )
    return val_means, uniforms, normals, lognormals

def get_config_mean(ind, means_, coef):
    return coef[0] + sum((cf[0] - cf[1]) * means_[ind][j] for j, cf in enumerate(coef[1]))

def get_response_aux(ind, cof, qtl):
    return sum(np.multiply(cof[t][0], qtl[ind][t][0]) + np.multiply(cof[t][1], qtl[ind][t][1]) for t in range(len(cof)))

def get_response():
    for current_coef, cnf in enumerate(coefficients):
        v = cnf[0] * np.ones(GridSize)
        cof = cnf[1]
        df_Unif_responses[f"Y_{current_coef}"] = [v + get_response_aux(ind, cof, qtl_uniforms) for ind in range(n_individuals)]
        df_Norm_responses[f"Y_{current_coef}"] = [v + get_response_aux(ind, cof, qtl_normals) for ind in range(n_individuals)]
        df_LogNorm_responses[f"Y_{current_coef}"] = [v + get_response_aux(ind, cof, qtl_lognormals) for ind in range(n_individuals)]

def get_config_str(confg):
    b, par = confg
    return f"$v={b}" + "".join(f",a_{i+1}={p[0]}, b_{i+1}={p[1]}" for i, p in enumerate(par)) + "$"

def get_test_variables(variables_ind, num_test, kFold):
    rd.shuffle(variables_ind)
    return [variables_ind[-num_test:] if len(variables_ind) > num_test else rd.sample(variables_ind, k=num_test) for _ in range(kFold)]

def run_simulation(sim):
    logging.info(f"==============Start Simulation ({sim})================== ")
    responses_variables = ["Y"]
    cols = ['Individual', 'Distribution'] + [f"X_data_{col+1}" for col in range(3)]
    data_Qf_unif = pd.DataFrame(columns=cols)
    data_Qf_normal = pd.DataFrame(columns=cols)
    data_Qf_log = pd.DataFrame(columns=cols)
    indi = [f"ind{i}" for i in range(n_individuals)]
    data_Qf_unif["Individual"], data_Qf_normal["Individual"], data_Qf_log["Individual"] = indi, indi, indi
    data_Qf_unif["Distribution"], data_Qf_normal["Distribution"], data_Qf_log["Distribution"] = ["Uniforms"] * n_individuals, ["Normals"] * n_individuals, ["LogNormals"] * n_individuals
    cols_df = ['Individual', 'Distribution'] + [f"X_{col+1}" for col in range(3)]
    data_unif_Qf = pd.DataFrame(columns=cols_df)
    data_normal_Qf = pd.DataFrame(columns=cols_df)
    data_log_Qf = pd.DataFrame(columns=cols_df)
    data_unif_Qf["Individual"], data_normal_Qf["Individual"], data_log_Qf["Individual"] = indi, indi, indi
    data_unif_Qf["Distribution"], data_normal_Qf["Distribution"], data_log_Qf["Distribution"] = ["Uniforms"] * n_individuals, ["Normals"] * n_individuals, ["LogNormals"] * n_individuals
    rd_factror = [rd.random() for _ in indi]
    erros_columns = ["Individual"]
    responses_columns = ["Individual"] + [f"Y_{cnf}" for cnf in range(len(coefficients))] + [f"Y_e_u_{cnf}{mu}{vu}" for cnf in range(len(coefficients)) for mu in range(len(factors)) for vu in range(len(var_factors))]
    data_err_unif_Qf, data_err_normal_Qf, data_err_log_Qf = pd.DataFrame(columns=erros_columns), pd.DataFrame(columns=erros_columns), pd.DataFrame(columns=erros_columns)
    df_Unif_responses, df_Norm_responses, df_LogNorm_responses = pd.DataFrame(columns=responses_columns), pd.DataFrame(columns=responses_columns), pd.DataFrame(columns=responses_columns)
    data_err_unif_Qf_all, data_err_normal_Qf_all, data_err_log_Qf_all = pd.DataFrame(columns=erros_columns), pd.DataFrame(columns=erros_columns), pd.DataFrame(columns=erros_columns)
    data_err_unif_Qf["Individual"], data_err_normal_Qf["Individual"], data_err_log_Qf["Individual"] = indi, indi, indi
    data_err_unif_Qf_all["Individual"], data_err_normal_Qf_all["Individual"], data_err_log_Qf_all["Individual"] = indi, indi, indi
    df_Unif_responses["Individual"], df_Norm_responses["Individual"], df_LogNorm_responses["Individual"] = indi, indi, indi
    logging.info("      Start - Get Synthectic Data ")
    means_, uniforms, normals, lognormals = generate_synthetic_data(n_individuals, num_data_points)
    logging.info("          End Distribution ")
    err_means = [[get_config_mean(ind, means_, cnf) for cnf in coefficients] for ind in range(n_individuals)]
    logging.info("          Start Quantiles ")
    qtl_uniforms = [[kde.get_quantiles(uniforms[ind][k]) for k in range(3)] for ind in range(n_individuals)]
    qtl_normals = [[kde.get_quantiles(normals[ind][k]) for k in range(3)] for ind in range(n_individuals)]
    qtl_lognormals = [[kde.get_quantiles(lognormals[ind][k]) for k in range(3)] for ind in range(n_individuals)]
    logging.info("          End Quantiles ")
    logging.info("          Start Erros ")
    err_uniforms, err_normals, err_lognormals = [], [], []
    for ind in range(n_individuals):
        curr_u, curr_n, curr_l = [], [], []
        for k in range(len(coefficients)):
            cf_u, cf_n, cf_l = [], [], []
            for mu in factors:
                m_u, m_n, m_l = [], [], []
                for vr in var_factors:
                    m_k = mu * err_means[ind][k]
                    v_k = vr * m_k
                    m_u.append(np.random.uniform(m_k - np.sqrt(3 * v_k), m_k + np.sqrt(3 * v_k), num_data_points))
                    m_n.append(np.random.normal(m_k, np.sqrt(v_k), num_data_points))
                    m_l.append(np.random.lognormal(0.5 * np.log((m_k)**4 / ((v_k) + (m_k)**2)), np.sqrt(np.log(1 + ((v_k) / (m_k)**2))), num_data_points))
                cf_u.append(m_u)
                cf_n.append(m_n)
                cf_l.append(m_l)
            curr_u.append(cf_u)
            curr_n.append(cf_n)
            curr_l.append(cf_l)
        err_uniforms.append(curr_u)
        err_normals.append(curr_n)
        err_lognormals.append(curr_l)
    logging.info("          End Errors ")
    logging.info("          Start Responses ")
    for k in range(len(coefficients)):
        for mu in range(len(factors)):
            for vr in range(len(var_factors)):
                data_err_unif_Qf_all[f"Y_e_{k}{mu}{vr}"] = [kde.get_quantiles(err_uniforms[ind][k][mu][vr]) for ind in range(n_individuals)]
                data_err_normal_Qf_all[f"Y_e_{k}{mu}{vr}"] = [kde.get_quantiles(err_normals[ind][k][mu][vr]) for ind in range(n_individuals)]
                data_err_log_Qf_all[f"Y_e_{k}{mu}{vr}"] = [kde.get_quantiles(err_lognormals[ind][k][mu][vr]) for ind in range(n_individuals)]
    rnd = [0 if rd.random() < thold else 1 for _ in range(n_individuals)]
    for k in range(len(coefficients)):
        for mu in range(len(factors)):
            for vr in range(len(var_factors)):
                data_err_unif_Qf[f"Y_e_{k}{mu}{vr}"] = [data_err_unif_Qf_all[f"Y_e_{k}{mu}{vr}"][i][rnd[i]] for i in range(n_individuals)]
                data_err_normal_Qf[f"Y_e_{k}{mu}{vr}"] = [data_err_normal_Qf_all[f"Y_e_{k}{mu}{vr}"][i][rnd[i]] for i in range(n_individuals)]
                data_err_log_Qf[f"Y_e_{k}{mu}{vr}"] = [data_err_log_Qf_all[f"Y_e_{k}{mu}{vr}"][i][rnd[i]] for i in range(n_individuals)]
    logging.info("          End Responses ")
    logging.info("      End - Get Synthectic Data ")
    logging.info("     Set error responses ")
    for k in range(len(coefficients)):
        for mu in range(len(factors)):
            for vr in range(len(var_factors)):
                df_Unif_responses[f"Y_e_u_{k}{mu}{vr}"] = [df_Unif_responses[f"Y_{k}"][ind] + data_err_unif_Qf[f"Y_e_{k}{mu}{vr}"][ind] for ind in range(n_individuals)]
                df_Unif_responses[f"Y_e_n_{k}{mu}{vr}"] = [df_Unif_responses[f"Y_{k}"][ind] + data_err_normal_Qf[f"Y_e_{k}{mu}{vr}"][ind] for ind in range(n_individuals)]
                df_Unif_responses[f"Y_e_l_{k}{mu}{vr}"] = [df_Unif_responses[f"Y_{k}"][ind] + data_err_log_Qf[f"Y_e_{k}{mu}{vr}"][ind] for ind in range(n_individuals)]
                df_Norm_responses[f"Y_e_u_{k}{mu}{vr}"] = [df_Norm_responses[f"Y_{k}"][ind] + data_err_unif_Qf[f"Y_e_{k}{mu}{vr}"][ind] for ind in range(n_individuals)]
                df_Norm_responses[f"Y_e_n_{k}{mu}{vr}"] = [df_Norm_responses[f"Y_{k}"][ind] + data_err_normal_Qf[f"Y_e_{k}{mu}{vr}"][ind] for ind in range(n_individuals)]
                df_Norm_responses[f"Y_e_l_{k}{mu}{vr}"] = [df_Norm_responses[f"Y_{k}"][ind] + data_err_log_Qf[f"Y_e_{k}{mu}{vr}"][ind] for ind in range(n_individuals)]
                df_LogNorm_responses[f"Y_e_u_{k}{mu}{vr}"] = [df_LogNorm_responses[f"Y_{k}"][ind] + data_err_unif_Qf[f"Y_e_{k}{mu}{vr}"][ind] for ind in range(n_individuals)]
                df_LogNorm_responses[f"Y_e_n_{k}{mu}{vr}"] = [df_LogNorm_responses[f"Y_{k}"][ind] + data_err_normal_Qf[f"Y_e_{k}{mu}{vr}"][ind] for ind in range(n_individuals)]
                df_LogNorm_responses[f"Y_e_l_{k}{mu}{vr}"] = [df_LogNorm_responses[f"Y_{k}"][ind] + data_err_log_Qf[f"Y_e_{k}{mu}{vr}"][ind] for ind in range(n_individuals)]
    logging.info("     Ended Set error responses ")
    variables_ind = [f"ind{i}" for i in range(n_individuals)]
    all_individuals = [f"ind{i}" for i in range(n_individuals)]
    rd.shuffle(all_individuals)
    small_sample_individuals = list(all_individuals[-n_small:])
    all_small_sample_individuals = list(all_individuals[-n_small:])
    logging.info("     Get error Sets ")
    test_variables = get_test_variables(all_individuals, n_test, kfoldRepeats)
    test_variables_small = get_test_variables(all_small_sample_individuals, n_test_small, kfoldRepeats)
    logging.info("     Ended Get error Sets ")
    print(f"######################## Start Fitting Models  ({sim}) ##################################")
    logging.info(f"     ######################## Start Fitting Models  ({sim}) ################################## ")
    try:
        uniform_jobs = []
        for distrib, data, dt_ind in zip(["Uniform", "Normal", "LogNormal"], [df_Unif_responses, df_Norm_responses, df_LogNorm_responses], [qtl_uniforms, qtl_normals, qtl_lognormals]):
            logging.info(f"     Distribution={distrib} ")
            logging.info(f"     small_sample_individuals={small_sample_individuals} ")
            smallData_Qf = data[data["Individual"].isin(small_sample_individuals)]
            smallData_Qf.reset_index(drop=True, inplace=True)
            for coef_simulation, coef in enumerate(coefficients):
                logging.info(f"          Start Coefcient {coef} ")
                cols = ["Individual"] + [f"X_{k+1}" for k in range(num_reg[coef_simulation])] + responses_variables
                co_variables = [f"X_{k+1}" for k in range(num_reg[coef_simulation])]
                df = pd.DataFrame(columns=cols)
                df['Individual'] = indi
                df_small = pd.DataFrame(columns=cols)
                df_small['Individual'] = small_sample_individuals
                for k in range(num_reg[coef_simulation]):
                    df[f"X_{k+1}"] = [dt_ind[j][k] for j in range(n_individuals)]
                    df_small[f"X_{k+1}"] = [dt_ind[j][k] for j in list([int(i.replace('ind', '')) for i in small_sample_individuals])]
                df["Y"] = data[f"Y_{coef_simulation}"]
                df_small["Y"] = data[f"Y_{coef_simulation}"]
                job = Process(target=get_simulation_response, args=(data_range, GridSize, sim, df, co_variables, kfoldRepeats, n_individuals, len(coef[1]), num_data_points, get_config_str(coef), indi, "Y", 'Individual', '', f'{distrib}', "", test_variables,))
                job.start()
                uniform_jobs.append(job)
                job1 = Process(target=get_simulation_response, args=(data_range, GridSize, sim, df_small, co_variables, 10, n_individuals_small, len(coef[1]), num_data_points, get_config_str(coef), small_sample_individuals, "Y", 'Individual', '', f'{distrib}', "", test_variables_small,))
                job1.start()
                uniform_jobs.append(job1)
                for er, dt_err in zip(["u", "n", "l"], [data_err_unif_Qf, data_err_normal_Qf, data_err_log_Qf]):
                    logging.info(f"     Start Jobs, error={er} ")
                    for u in range(len(factors)):
                        for v in range(len(var_factors)):
                            df["Y"] = data[f"Y_e_{er}_{coef_simulation}{u}{v}"]
                            job3 = Process(target=get_simulation_response, args=(data_range, GridSize, sim, df, co_variables, 10, n_individuals, len(coef[1]), num_data_points, get_config_str(coef), indi, "Y", 'Individual', er, f'{distrib}', f"{factors[u]}|{var_factors[v]}", test_variables,))
                            job3.start()
                            uniform_jobs.append(job3)
                            job4 = Process(target=get_simulation_response, args=(data_range, GridSize, sim, df_small, co_variables, 10, n_individuals_small, len(coef[1]), num_data_points, get_config_str(coef), small_sample_individuals, "Y", 'Individual', er, f'{distrib}', f"{factors[u]}|{var_factors[v]}", test_variables_small,))
                            job4.start()
                            uniform_jobs.append(job4)
            for job in uniform_jobs:
                job.join()
    except Exception as exception:
        input(exception)
        logging.error(f"Exception occurred: {exception}", exc_info=True)
        input()
    logging.info(f"######################## Completed Fitting Models  ({sim}) ##################################")

if __name__ == '__main__':
    n_simul = int(input("How Many Simulations? "))
    kfoldRepeats = 10
    num_data_points = 5000
    n_individuals=250
    GridSize=1000
    data_range = [0.001, 1]
    # Initialize KDE
    thold = 0.5
    kde = Kde.KernelDensities(GridSize, thold)
    # Convert lists to NumPy arrays for element-wise operations
    mean_var = np.array([20, 10, 5])
    sigmas = np.array([8, 6, 4])
    dist_coef = [0.7, 1.3]
    
    coefficients  =[
        [-10,[[4,1]]],
        [30,[[2,3]]],
        [-25,[[8,4]]],
        [-10,[[2,1],[3.5,1],[1,3.5]]],
        [10,[[10,2],[5,9],[10,20]]]
    ]
    factors=[0.05,.30]
    var_factors=[0.05,.30]
    dist_coef = [0.7,1.3] 
    num_reg=[1,1,1,3,3]
    n_small=int(n_individuals/5)
    n_individuals_small=n_small
    n_test=int(n_individuals*0.2)
    n_test_small=int(n_small*0.2) 
    kde = Kde.KernelDensities(GridSize,thold) 