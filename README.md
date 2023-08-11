# Building and Construction Industry Security of Payment Adjudication Analysis 🏗️
### Main Contributor: Justin Lee
### Credit to Other Contributors: Angus Crawshaw, Saskia Ritman, Caroline Lei 

## Introduction of Intended Use for Code and CSV Data Files 📊
This project explores and analyzes the Building and Construction Industry Security of Payment (SOP) adjudication data to understand and predict dispute outcomes and adjudication amounts. The code and data files serve the following purposes:

* **Preprocessing and Wrangling**: Data from `adjudication.csv`, `australian_postcodes.csv`, and `post_code_data.csv` is cleaned and transformed for analysis.
* **K-Nearest Neighbors (KNN) Prediction**: Utilizes KNN to predict determination status of a claim using the mentioned datasets.
* **Linear Regression Analysis**: Applies linear regression to forecast the adjudication amount based on the claimed amount in `adjudication.csv`.
* **Visualization**: Generates various visual representations to provide insights into the results.

### Data Files 📁
* `adjudication.csv`: Contains data related to claims made to the VBA based on the SOP Act. [Source](https://discover.data.vic.gov.au/dataset/building-and-construction-industry-security-of-payment-adjudication-activity-data)
* `australian_postcodes.csv`: Includes information on Australian postcodes. [Source](https://www.matthewproctor.com/full_australian_postcodes_vicd)
* `post_code_data.csv`: Encompasses socioeconomic status data of Australian postcodes. [Source](https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs-edition-3/jul2021-jun2026/non-abs-structures/postal-areas)

## File Structure and Use of Each File 📑
### Script Files
1. `1a_data_wrangling.py`: Performs preprocessing and wrangling of the adjudication dataset.
2. `1b_normal_distribution_claimedamount.py`: Produces a normal distribution plot of claimed amounts.
3. `2_corr_matrix.py`: Generates a correlation matrix based on features of the adjudication dataset.
4. `3_knn.py`: Implements the KNN supervised learning model for prediction.
5. `4_lin_regression.py`: Deploys a linear regression supervised learning model for forecasting.

### Data Files
* `adjudication.csv`: Primary dataset for analysis.
* `australian_postcodes.csv` and `post_code_data.csv`: Support datasets for in-depth analysis.

## Instructions on How to Run Your Code 💻
Execute the following code in Python, replacing "[document name]" with script file names in the order listed above:
```bash
python [document name]
```

## Additional Requirements to Run Code 🧩
* **Python Version**: 3.x
* **Libraries Used**:
    * Pandas
    * numPy
    * sklearn
    * matplotlib
    * seaborn
    * re


#Add table
#Add thing where you can enter emperical data and it returns yes or no and why
#Add the video 

[Click here to view the video](hyperparam_obsession.mov)
