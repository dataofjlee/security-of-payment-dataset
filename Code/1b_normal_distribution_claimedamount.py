import re
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

# Load the data
adjudication_data = pd.read_csv('adjudication.csv')

# Clean the data
claimed_amount = adjudication_data['Claimed amount (ex GST)']
claimed_amount = claimed_amount.str.replace(',', '').astype(float)

# Apply logarithm to 'claimed_amount'
claimed_amount = np.log(claimed_amount)
print(claimed_amount)
print("\nMean logarithm: ", np.mean(claimed_amount))
print("Plot generated ✅")

# Set up the style and size of the plot
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))

# Create the histogram using Seaborn
sns.histplot(claimed_amount, bins=30, kde=True, color='skyblue', linewidth=0.5, edgecolor='black')

# Set the title and labels
plt.title('Histogram of Natural Log of Claimed Amount (ex GST)', fontsize=15)
plt.xlabel('Natural Log of Claimed Amount (ex GST)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)

# Show the plot
plt.show()

# Save the figure
plt.savefig('NormalDistributionClaimedAmount.png')
