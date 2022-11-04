import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm

crypto = pd.read_excel('./Data.xlsx', 'Crypto', index_col=0)  # Step1
factor = pd.read_excel('./Data.xlsx', 'Factor', index_col=0)
crypto_ret = np.array(np.log(crypto / crypto.shift()).dropna(how='all'))  # Log return
fact_ret = np.array(np.log(factor / factor.shift()).dropna(how='all'))

reqExp = 0.9  # We set the required explanatory power as 0.9
reqCorr = 0.2  # We set the required minimum correlation for the factor with the eigen portfolio as 0.2
reqFcorr = 0.7  # We set the maximum allowed between-factor correlation as 0.7

# Perform a PCA analysis on the factors using numpy linalg.eig, find the minimum number of
# principal components to cover the required explanatory power.
crypto_cov = np.cov(crypto_ret.T)
eig_values, eig_vectors = np.linalg.eig(crypto_cov)
cum_ex_pow = np.cumsum(eig_values/eig_values.sum())  # cumulative explanatory power
min_PC = np.argmax(cum_ex_pow > reqExp) + 1  # Find out in which cumulative sum > required explanatory power
print(f"The minimum number of principal components to cover the required explanatory power ({reqExp}) is {min_PC}.")

# This is to find the most relevant factors to represent the principal components.
# We set up some empty lists to store the factor names, factor correlation.
factor_names = []
factor_correlation = []
factor_col = []

# The algorithm as follow:
# PC1: run each factor correlation with PC1 (pearsonr from scipy.stats). For the first factor, if the
# correlation (absolute) is greater than ‘reqCorr’, keep it. For the 2nd factor onward, the correlation
# needs to be greater than ‘reqCorr’ but less than the ‘reqFcorr’ to be kept.
# After PC1, we must have some factors in the list already so we go on for PC2 and then PC3: For each
# factor, keep those with correlation greater than ‘reqCorr’ but less than the ‘reqFcorr’.
crypto_pca = np.dot(crypto_ret, (eig_vectors.T[:][:min_PC]).T)  # Project data onto lower-dimensional linear subspace
for i in np.arange(min_PC):
    for j in np.arange(len(fact_ret[0])):
        fact_matrix = fact_ret.T[j]
        crypto_corr = stats.pearsonr(fact_matrix.T, crypto_pca.T[i])[0]
        if (len(factor_names) == 0) & (abs(crypto_corr) > reqCorr):
            factor_correlation.append(crypto_corr)
            factor_names.append(factor.columns[j])
            factor_col.append(j)
        if (len(factor_names) != 0) & (abs(crypto_corr) > reqCorr) & (factor.columns[j] not in factor_names):
            check_Fcorr = 0
            for factors in factor_names:
                F = fact_ret.T[factor.columns.get_loc(factors)]
                Fcorr = stats.pearsonr(fact_matrix.T, F)[0]
                if abs(Fcorr) > reqFcorr:
                    check_Fcorr = 1
                    break
            if check_Fcorr == 0:
                factor_correlation.append(crypto_corr)
                factor_names.append(factor.columns[j])
                factor_col.append(j)
factor_list = pd.DataFrame(data={'Names': factor_names, 'Correlation': factor_correlation})
factor_list_sorted = factor_list.reindex(factor_list.Correlation.abs().sort_values(ascending=False).index).reset_index(
    drop=True)
print(factor_list_sorted)

# With the list of factors from Q4, we normalize their returns and crypto indexes as well.
crypto_Nret = (crypto_ret - np.mean(crypto_ret, 0)) / np.std(crypto_ret, 0, ddof=1)  # Step5
fact_Nret = (fact_ret - np.mean(fact_ret, 0)) / np.std(fact_ret, 0, ddof=1)

# Run a for loop for each equity index over the standardized factors from Q4: the OLS (statsmodels.api) with
# intercept. Then, we retrieve the beta, t-value and the Rsquare and keep them into 3 different list.
beta = pd.DataFrame(index=[['Intercept'] + factor_names])  # Step6
t_value = pd.DataFrame(index=[['Intercept'] + factor_names])
r_square = pd.DataFrame(index=['Rsquared'])
X = fact_Nret.T[factor_col]
X = sm.add_constant(X.T)
for i in np.arange(len(crypto.columns)):
    Y = crypto_Nret.T[i]
    results = sm.OLS(Y, X).fit()
    beta[crypto.columns[i]] = results.params
    t_value[crypto.columns[i]] = results.tvalues
    r_square[crypto.columns[i]] = results.rsquared
beta.to_csv('beta.csv')
t_value.to_csv('tvalue.csv')
r_square.to_csv('Rsq.csv')