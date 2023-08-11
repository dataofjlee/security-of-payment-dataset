import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

# Can we predict ' Adjudicated amount (ex GST) '  using Linear Regression
adjudication_data = pd.read_csv('adjudication.csv')
cleaned_dataset = pd.read_csv('cleaned_dataset_final.csv')

cleaned_dataset['adjudicated_amount'] = adjudication_data[" Adjudicated amount (ex GST) "]
cleaned_dataset = cleaned_dataset[~cleaned_dataset['adjudicated_amount'].isin(['N/A', 'Withheld', 'Not yet determined', '0.00'])].dropna()
cleaned_dataset['adjudicated_amount'] = cleaned_dataset['adjudicated_amount'].str.replace(',', '')
cleaned_dataset['adjudicated_amount'] = cleaned_dataset['adjudicated_amount'].astype(float)
cleaned_dataset['adjudicated_amount'] = np.log(cleaned_dataset['adjudicated_amount'] + 0.1)  # For 0 log

# Fit and transform 'adjudicated_amount' using scaler_adj
scaler_adj = StandardScaler()
cleaned_dataset['adjudicated_amount'] = scaler_adj.fit_transform(cleaned_dataset['adjudicated_amount'].values.reshape(-1, 1))

# Fit and transform 'claimed_amount' using scaler_claim
duplicate_claimed = (adjudication_data['Claimed amount (ex GST)'].str.replace(',', '').astype(float))
duplicate_claimed = np.log(duplicate_claimed)
scaler_claim = StandardScaler()
duplicate_claimed = scaler_claim.fit_transform(duplicate_claimed.values.reshape(-1, 1))

bar = "+=======================================================+"


# Calculate correlation matrix
correlation_matrix = cleaned_dataset.corr()

# Now calculate top 10 correlation to adjudicated amount using correlation matrix 
print()
print(bar)
print("HIGHEST CORRELATED TO ADJUDICATED AMOUNT:\n", correlation_matrix['adjudicated_amount'].abs().sort_values(ascending=False)[1:11], sep='')
print(bar)
print()

# Create the linear regression model
linear_reg = LinearRegression()

# Perform cross-validation with 10 partitions
scores = cross_val_score(linear_reg, cleaned_dataset[['determination_status', 'acceptance_date_isNA', 'extension_of_time_sought_Yes', 'notice_sent_to_claimant_Yes', 'claimant_advisers_None', 'claimant_advisers_Solicitors', 'claimed_amount']], cleaned_dataset['adjudicated_amount'], cv=10, scoring='neg_mean_squared_error')

# Convert scores to positive values
mse_scores = -scores

# Calculate mean and standard deviation of MSE scores
mean_mse = mse_scores.mean()
std_mse = mse_scores.std()

# Print evaluation metrics
print()
print(bar)
print("Cross-Validation Evaluation Metrics:")
print("Mean Squared Error (MSE):", mean_mse)
print("Root Mean Squared Error (RMSE):", np.sqrt(mean_mse))
print("Standard Deviation (MSE):", std_mse)
print(bar)
print()

# 80:20 TEST SPLIT
x_train, x_test, y_train, y_test = train_test_split(cleaned_dataset[['determination_status', 'acceptance_date_isNA', 'extension_of_time_sought_Yes', 'notice_sent_to_claimant_Yes', 'claimant_advisers_None', 'claimant_advisers_Solicitors', 'claimed_amount']], cleaned_dataset['adjudicated_amount'], test_size=0.2, random_state=42)
linear_reg = LinearRegression()
linear_reg.fit(x_train, y_train)
y_pred = linear_reg.predict(x_test)

# Reverse standardization
# Reverse log transformation
original_values_x = np.exp(scaler_claim.inverse_transform(x_test['claimed_amount'].values.reshape(1, -1)))
original_values_x = np.squeeze(original_values_x)

original_values_y = np.exp(scaler_adj.inverse_transform(y_test.values.reshape(-1, 1)))-0.1
original_values_y = np.squeeze(original_values_y)

descaled_predicted = np.exp(scaler_adj.inverse_transform(y_pred.reshape(-1, 1)))-0.1
descaled_predicted = np.squeeze(descaled_predicted)

#print("T_test\n", original_values)
#print("Original Claim Amounts:", adjudication_data['Claimed amount (ex GST)'].loc[x_test['claimed_amount'].index])

print()
print(bar)
print("Example Prediction Comparisons by L.R Model: ")
for i in range(5 if 5 < len(x_test) else len(x_test)):
    print(f"Claimed amount: ${original_values_x[i]:.2f}, Predicted adjudication amount: ${descaled_predicted[i]:.2f}, True adjudication amount: ${original_values_y[i]:.2f}")
print(bar)
print()

# Create a copy of x_train with encoded categorical variables if necessary
x_train_encoded = x_train.copy()

# Set style and context for better visualization
sns.set(style='whitegrid', font_scale=1.2)

# Partial Dependence Plots
fig, axs = plt.subplots(2, 4, figsize=(16, 8))
fig.subplots_adjust(hspace=0.4)

features = ['determination_status', 'acceptance_date_isNA', 'extension_of_time_sought_Yes', 'notice_sent_to_claimant_Yes', 'claimant_advisers_None', 'claimant_advisers_Solicitors', 'claimed_amount']
for i, feature in enumerate(features):
    row, col = divmod(i, 4)
    sns.regplot(x=x_train_encoded[feature], y=y_train, ax=axs[row, col], scatter_kws={'alpha': 0.5})
    axs[row, col].set_xlabel(feature, fontsize=12)
    axs[row, col].set_ylabel('Adjudicated Amount', fontsize=12)
    axs[row, col].tick_params(axis='both', which='major', labelsize=10)
    axs[row, col].spines['top'].set_visible(False)
    axs[row, col].spines['right'].set_visible(False)
    axs[row, col].spines['bottom'].set_linewidth(0.5)
    axs[row, col].spines['left'].set_linewidth(0.5)

plt.tight_layout()
plt.savefig('linear_regression_plot.png', dpi=300)


# LIVE Testing
# FIRST retrain model on entire dataset 
linear_reg_full = LinearRegression()
linear_reg_full.fit(cleaned_dataset[['determination_status', 'acceptance_date_isNA', 'extension_of_time_sought_Yes', 'notice_sent_to_claimant_Yes', 'claimant_advisers_None', 'claimant_advisers_Solicitors', 'claimed_amount']], cleaned_dataset['adjudicated_amount'])

# User Inputs
x_user = pd.DataFrame()

while True:
    user_claimed_amount = float(input("Enter the claimed amount ($AUD): "))
    if 0 < user_claimed_amount and 50_000_000 > user_claimed_amount:
        break

determination_status = input("Was the determination status determined? (Y/N): ")
x_user['determination_status'] = [1 if determination_status.upper() == 'Y' else 0]

acceptance_date_isNA = input("Is the acceptance date missing? (Y/N): ")
x_user['acceptance_date_isNA'] = [1 if acceptance_date_isNA.upper() == 'Y' else 0]

extension_of_time_sought = input("Did the claimant seek an extension of time? (Y/N): ")
x_user['extension_of_time_sought_Yes'] = [1 if extension_of_time_sought.upper() == 'Y' else 0]

notice_sent_to_claimant = input("Was a notice sent to the claimant? (Y/N): ")
x_user['notice_sent_to_claimant_Yes'] = [1 if notice_sent_to_claimant.upper() == 'Y' else 0]

claimant_advisers_none = input("Did the claimant have no advisers? (Y/N): ")
x_user['claimant_advisers_None'] = [1 if claimant_advisers_none.upper() == 'Y' else 0]

claimant_advisers_solicitors = input("Did the claimant have solicitors as advisers? (Y/N): ")
x_user['claimant_advisers_Solicitors'] = [1 if claimant_advisers_solicitors.upper() == 'Y' else 0]

x_user['claimed_amount'] = [user_claimed_amount]
x_user['claimed_amount'] = scaler_claim.fit_transform(x_user['claimed_amount'].values.reshape(-1, 1)) 

y_pred_user = linear_reg_full.predict(x_user)
y_pred_user = np.exp(scaler_adj.inverse_transform(y_pred_user.reshape(-1, 1)))-0.1
y_pred_user = np.squeeze(y_pred_user)
print()
print(f"Predicted Adjudicated Amount: ${y_pred_user:.2f}")

