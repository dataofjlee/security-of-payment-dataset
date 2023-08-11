import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Read in data
adjudication_data = pd.read_csv('adjudication.csv')
postcode_data = pd.read_csv('post_code_data.csv')

'''
DATA PREPROCESSING
Only features which precede determination status will be including, 
representing factors that are known before the claim is made
'''

# Initialise
cleaned_dataset = pd.DataFrame()

# The target feature
cleaned_dataset['determination_status'] = [1 if status == "Determined" else 0 for status in adjudication_data['Determination status']]

'''
Dates: need to be in numerical format, so converted to days of week or days since
'''

# Convert Application Date/Acceptance Dates + Create new Features: day of week, convert all N/A to NaT (Not a Time)
cleaned_dataset['app_date_day_of_week'] = pd.to_datetime(adjudication_data['Application date'], errors='coerce').dt.dayofweek
# Normalise
cleaned_dataset['app_date_day_of_week'] = cleaned_dataset['app_date_day_of_week'] / 7

# Fill N/A with imputed values, but first create a new column 'is_NA' to capture any significance of NA
cleaned_dataset['acceptance_date_isNA'] = [1 if pd.isna(date) else 0 for date in adjudication_data['Acceptance date']]

cleaned_dataset['acc_date_day_of_week'] = pd.to_datetime(adjudication_data['Acceptance date'], errors='coerce').dt.dayofweek
cleaned_dataset['acc_date_day_of_week'].fillna(cleaned_dataset['acc_date_day_of_week'].mean(), inplace=True)
cleaned_dataset['acc_date_day_of_week'] = cleaned_dataset['acc_date_day_of_week'] / 7

cleaned_dataset['time_to_acceptance'] = (pd.to_datetime(adjudication_data['Acceptance date'], errors='coerce')-pd.to_datetime(adjudication_data['Application date'], errors='coerce')).dt.days
cleaned_dataset['time_to_acceptance'].fillna(cleaned_dataset['time_to_acceptance'].mean(), inplace=True)
cleaned_dataset['time_to_acceptance'] = (cleaned_dataset['time_to_acceptance'] - cleaned_dataset['time_to_acceptance'].min()) / (cleaned_dataset['time_to_acceptance'].max() - cleaned_dataset['time_to_acceptance'].min())

'''
Discretise/Encode Categorical Variables: description, post code, payment schedule provision, section, business type, determination status, etc.
Filter out some categorical features with contextual judgement of relevancy
OR features with low variance were also eliminated 
Used one-hot encoding.
'''

cleaned_dataset['payment_sched_provided'] = [1 if value=="Yes" else 0 for value in adjudication_data['Payment Schedule provided']]

encoded_columns = pd.get_dummies(adjudication_data['Business Type/Activity (Claimant)'], prefix='business_type_claimant')
cleaned_dataset = pd.concat([cleaned_dataset, encoded_columns], axis=1)

encoded_columns = pd.get_dummies(adjudication_data['Business Structure (Claimant)'], prefix='business_structure_claimant')
cleaned_dataset = pd.concat([cleaned_dataset, encoded_columns], axis=1)

encoded_columns = pd.get_dummies(adjudication_data['Claimant advisers'], prefix='claimant_advisers')
cleaned_dataset = pd.concat([cleaned_dataset, encoded_columns], axis=1)

encoded_columns = pd.get_dummies(adjudication_data['Business Type/Activity (Respondent)'], prefix='business_type_respondent')
cleaned_dataset = pd.concat([cleaned_dataset, encoded_columns], axis=1)

encoded_columns = pd.get_dummies(adjudication_data['Business Structure (Respondent)'], prefix='business_structure_respondent')
cleaned_dataset = pd.concat([cleaned_dataset, encoded_columns], axis=1)

encoded_columns = pd.get_dummies(adjudication_data['Respondent advisers'], prefix='respondent_advisers')
cleaned_dataset = pd.concat([cleaned_dataset, encoded_columns], axis=1)

encoded_columns = pd.get_dummies(adjudication_data['s21(2B) notice sent to Claimant'], prefix='notice_sent_to_claimant')
cleaned_dataset = pd.concat([cleaned_dataset, encoded_columns], axis=1)

encoded_columns = pd.get_dummies(adjudication_data['s22(4)(b) extension of time sought'], prefix='extension_of_time_sought')
cleaned_dataset = pd.concat([cleaned_dataset, encoded_columns], axis=1)

'''
Prices:
Convert $Amount and Float Columns to appropriate float representation, then scale using standardisation and normalisation
Take log of prices so they are normally distributed and can be standardised 
For claimed_amount no imputation needed 
'''

cleaned_dataset['claimed_amount'] = (adjudication_data['Claimed amount (ex GST)'].str.replace(',', '').astype(float))
cleaned_dataset['claimed_amount'] = np.log(cleaned_dataset['claimed_amount'])
scaler = StandardScaler()
cleaned_dataset['claimed_amount'] = scaler.fit_transform(cleaned_dataset['claimed_amount'].values.reshape(-1, 1))
cleaned_dataset['claimed_amount'].fillna(0, inplace=True)

'''
Postcode processing: postcodes in '3124' form are feature engineered into: 
Affluency score (IRSAD) and categorised into their general regions 
Dataset: Full Australian Postcodes VIC
Source: Matthew Proctor doctor proctor lol
Website: https://www.matthewproctor.com/full_australian_postcodes_vic
'''

# Postcode Data on Affluency
postcode_data = pd.read_csv('post_code_data.csv')

# Postcode data on general regions 
more_postcode_data = pd.read_csv('australian_postcodes.csv')
more_postcode_data = more_postcode_data.loc[(more_postcode_data['postcode'] >= 3000) & (more_postcode_data['postcode'] < 4000)].reset_index(drop=True)
more_postcode_data['region'] = more_postcode_data['region'].str.replace('R', '')
region_dict = more_postcode_data.set_index('postcode')['region'].to_dict()

# Remove postal codes not in victoria
postcode_data = postcode_data.loc[(postcode_data['2016 Postal Area (POA) Code'] >= 3000) & (postcode_data['2016 Postal Area (POA) Code'] < 4000)]
postcode_data.rename(columns={'Index of Relative Socio-economic Advantage and Disadvantage': 'IRSAD Score', 'Unnamed: 4': 'IRSAD Decile'}, inplace=True)

# Convert postcodes into integer
postcode_series = pd.to_numeric(postcode_data['2016 Postal Area (POA) Code'], downcast='integer')

# Min-Max Normalises the IRSAD Scores 
scaler = StandardScaler()
postcode_data['IRSAD Score'] = postcode_data['IRSAD Score'].astype(float)
#postcode_data['IRSAD Score'] = (postcode_data['IRSAD Score'] - postcode_data['IRSAD Score'].min()) / (postcode_data['IRSAD Score'].max() - postcode_data['IRSAD Score'].min())
postcode_data['IRSAD Score Norm'] = scaler.fit_transform(postcode_data['IRSAD Score'].values.reshape(-1, 1))
postcode_data['IRSAD Decile'] = postcode_data['IRSAD Decile'].astype(int)

# Convert df to dict
postcode_dict = postcode_data.set_index('2016 Postal Area (POA) Code')['IRSAD Score Norm'].to_dict()
decile_dict = postcode_data.set_index('2016 Postal Area (POA) Code')['IRSAD Decile'].to_dict()

adjudication_postcode = adjudication_data['Project postcode'].replace('Multiple', '3169').astype(int)
decile_column = adjudication_data['Project postcode'].replace('Multiple', '3169').astype(int)
region_column = adjudication_data['Project postcode'].replace('Multiple', '3169').astype(int)

for i, postcode in enumerate(adjudication_postcode):
    if postcode in postcode_dict:
        adjudication_postcode[i] = postcode_dict[postcode]
        decile_column[i] = (decile_dict[postcode])/10
        region_column[i] = region_dict[postcode]
    else:
        adjudication_postcode[i] = postcode_data['IRSAD Score Norm'].mean()
        decile_column[i] = 5/10
        region_column[i] = more_postcode_data['region'].mean()

# for the regions get the dummy variables
encoded_columns = pd.get_dummies(region_column, prefix='claim_region')
cleaned_dataset = pd.concat([cleaned_dataset, encoded_columns], axis=1)

cleaned_dataset['irsad_score'] = adjudication_postcode
cleaned_dataset['decile_score'] = decile_column

'''
Job description categorising using simple keyword string preprocessing
''' 
def categorise(description):
    description = description.lower()
    if "construction" in description or "concreting" in description or "plumbing" in description or "roofing" in description or "carpentry" in description:
        return "Construction"
    elif "design" in description or "engineering" in description:
        return "Design"
    elif "supply" in description or "provision" in description:
        return "Supply"
    else:
        return "Other"

# Apply the function to the DataFrame
adjudication_data['Category'] = adjudication_data['Description of project and contract works'].apply(categorise)
encoded_columns = pd.get_dummies(adjudication_data['Category'], prefix='category')
cleaned_dataset = pd.concat([cleaned_dataset, encoded_columns], axis=1)

'''
Finished
Save to .csv
'''

# Export to .csv
cleaned_dataset.to_csv('cleaned_dataset_final.csv', index=False)
print("Saved to .csv")
print("Cleaned Dataset:\n")
print(cleaned_dataset.head())

# Splices first row for visualisation (ignore)
before_first_row = adjudication_data[:1]
after_first_row = cleaned_dataset[:1]
before_first_row.to_csv('before_first_row.csv', index=False)
after_first_row.to_csv('after_first_row.csv', index=False)
