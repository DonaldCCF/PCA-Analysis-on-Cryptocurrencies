# PCA-Analysis-on-Cryptocurrencies

(1) We use Pandas to read Data.xlsx into Dataframes ‘crypto’ and ‘factor’. Convert them into numpy arrays and construct the corresponding simple returns.
(Data are weekly closing price from 2/9/2018 to 11/4/2022)

Crypto indexes: 10 Cryptocurrnies <br/>
![image](https://user-images.githubusercontent.com/117000928/199917630-d4e55a68-5a69-4fde-9200-a3894811cc2e.png)

Factors such as Currencies, Stock Indexes, Commodities, CDS indexes <br/>
![image](https://user-images.githubusercontent.com/117000928/199918164-f149edab-23c3-4de4-baf8-0c8ae3ce6f84.png)

(2) We set the following parameters: The required explanatory power ‘reqExp’ as 0.8; the required minimum correlation for the factor with the eigen portfolio ‘reqCorr’ as 0.2; the maximum allowed between-factor correlation ‘reqFcorr’ as 0.7.

(3) Then, we perform a PCA analysis on the factors using numpy linalg.eig, find the minimum number of principal components to cover the required explanatory power.

Result:<br/> ![image](https://user-images.githubusercontent.com/117000928/199918804-b4d61e4e-a12e-4970-be18-9f1dffc41d2f.png)

(4) We want to find the most relevant factors to represent the principal components.
The algorithm as follow:
PC1: run each factor correlation with PC1 (pearsonr from scipy.stats). For the first factor, if the correlation (absolute) is greater than ‘reqCorr’, keep it. For the 2nd factor onward, the correlation needs to be greater than ‘reqCorr’ but less than the ‘reqFcorr’ to be kept. After PC1, you must have some factors in the list already, go on for PC2 and then PC3: For each factor, keep those with correlation greater than ‘reqCorr’ but less than the ‘reqFcorr’.

Result: <br/>![image](https://user-images.githubusercontent.com/117000928/199918966-c59961cb-9ade-4228-ab44-b9580b72c9ad.png) <br/>
The top three factors are S&P/ASX 200 Index,S&P 500 Index, FTSE 100 Index. However, the highest correlation for the factor with the eigen portfolio is 0.3 which is not able to fully explain.

(5) With the list of factors from (4), we normalize their returns and crypto indexes as well.

(6) Last, we Run a for loop for each crypto index over the standardized factors from (5): the OLS (statsmodels.api) with intercept (‘add_constant’ function), retrieve the beta, t-value and the Rsquare and keep them into 3 different list.

Beta <br/>
![image](https://user-images.githubusercontent.com/117000928/199920131-72e22c42-3cca-46fc-a95a-fe2557a29aa2.png)

T_value <br/>
![image](https://user-images.githubusercontent.com/117000928/199920262-d0dade07-bc17-4f4a-9e3e-fb5482ce171c.png)

Rsquare <br/>
![image](https://user-images.githubusercontent.com/117000928/199920969-88d98564-a56f-4d5d-812c-981527d47026.png)



