import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import LeaveOneOut, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import time
import seaborn as sns
import matplotlib.pyplot as plt

NUM_RECORDS = 324
MI_GRAPHS_NUM_COLUMNS = 6
BAR_COLOUR = (0/255, 181/255, 226/255)

cleaned_dataset = pd.read_csv('cleaned_dataset_final.csv')

start_time = time.time()

total_mi = {}

def get_filtered_features_mi(x_train, y_train, threshold=0.1):  
    # Calculate mutual information between features and target variable 'determination status'
    filtered_features = []

    mi_array = mutual_info_classif(X=x_train, y=y_train)
    for feature, mi in zip(cleaned_dataset.columns, mi_array):
        if mi > threshold:
            filtered_features.append(feature)

        if feature not in total_mi.keys():
            total_mi[feature] = mi
        else:
            total_mi[feature] = total_mi[feature] + mi

    return filtered_features

def get_best_k(k_x_train, k_y_train, k_x_test, k_y_test, k_values):
    # Defaults to 1
    best_k = 1
    best_accuracy = 0.0
    
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(k_x_train, k_y_train)
        y_pred = knn.predict(k_x_test)
        accuracy = metrics.accuracy_score(k_y_test, y_pred)
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k
    if best_accuracy==0.0:
        print("No k found")
    return best_k

# Step 4: Perform Leave-One-Out Cross Validation
loo = LeaveOneOut()
num_partitions = len(cleaned_dataset)

# Model Training and Prediction Metrics
num_truePositive = 0
num_falsePositive = 0
num_trueNegative = 0
num_falseNegative = 0

# Feature appearance dictionairy
appearances = dict()

# For each partition:
for train_index, test_index in loo.split(cleaned_dataset):
    progress = int(test_index[0] / (num_partitions-1) * 20)
    print("["+("✅" * (progress)) + ("❌" * (20-progress))+"]")
    print(f"TRAIN/TEST PARTITION: {test_index[0]} | SUCCESSFUL")

    # Define x/y test/train 
    x_train = cleaned_dataset.iloc[train_index].drop(columns=['determination_status'])
    y_train = cleaned_dataset.iloc[train_index]['determination_status']

    x_test = cleaned_dataset.iloc[test_index].drop(columns=['determination_status'])
    y_test = cleaned_dataset.iloc[test_index]['determination_status']
    
    # Determine features with MI value > threshold
    filtered_features = get_filtered_features_mi(x_train, y_train)
    
    for feature in filtered_features:
        if feature in appearances:
            appearances[feature] += 1
        else:
            appearances[feature] = 1
    

    # Determine the best k value by doing 80:20 split on x_train and x_test
    k_x_train, k_x_test, k_y_train, k_y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
    k_values = [k for k in range(1, 22, 2)]
    k = get_best_k(k_x_train, k_y_train, k_x_test, k_y_test, k_values)
    
    print(f"Hyperparameter k: {k}")

    # Now train and test on the 323 : 1 split 
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    
    # Evaluate whether true/false positive/negative and tally accordingly 
    print(f"Model predicted: {y_pred[0]} | True value: {y_test.values[0]}")
    if (y_pred[0]==1 and y_test.values[0]==1):
        num_truePositive += 1
    elif (y_pred[0]==1 and y_test.values[0]==0):
        num_falsePositive += 1
    elif (y_pred[0]==0 and y_test.values[0]==0):
        num_trueNegative += 1
    else:
        num_falseNegative += 1

    print()

# Calculate performance metrics
end_time = time.time()
runtime = end_time - start_time

accuracy = (num_truePositive + num_trueNegative) / (num_truePositive + num_trueNegative + num_falseNegative  + num_falsePositive)
precision = (num_truePositive) / (num_truePositive + num_falsePositive)
recall = (num_truePositive) / (num_truePositive + num_falseNegative)
f1_score = 2 * (precision * recall) / (precision + recall) 

bar = "+=======================================================+"
# Print or use the average metrics as needed
print(bar)
print("EVALUATION METRICS")
print(f"Runtime:  {runtime:>7.2f} seconds")
print(f"->  {round(runtime / num_partitions, 2)}s per partition")
print()
print(f"Accuracy:  {accuracy:>7.3f}")
print(f"Precision: {precision:>7.3f}")
print(f"Recall:    {recall:>7.3f}")
print(f"F1 Score:  {f1_score:>7.3f}")
print(bar)
print()

# Generate and save confusion matrix
confusion_matrix = pd.DataFrame(
    {
        'Predicted Positive': [num_truePositive, num_falsePositive],
        'Predicted Negative': [num_falseNegative, num_trueNegative]
    },
    index=['Actual Positive', 'Actual Negative']
)

plt.figure(figsize=(10,7))
plt.title('Confusion Matrix of KNN Model Performance')
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
print()
print(bar)
print("CONFUSION MATRIX:")
print(confusion_matrix)
print(bar)
print()

# Save the plot as a file
plt.savefig('confusion_matrix_knn.png')
plt.clf()

# Top 5 Features
print()
print(bar)
print("TOP 5 MI FEATURES:\n")
# Convert the dictionary to a list of tuples
appearance_list = list(appearances.items())

# Sort the list in descending order based on the number of appearances
appearance_list.sort(key=lambda x: x[1], reverse=True)

# Print the top 5 features
for feature, count in appearance_list[:5]:
    print(f"{feature:30} had {count:3} appearances")

sorted_appearances_tuples = sorted(appearances.items(), key = lambda x:x[1], reverse=True)
sorted_appearances = dict(sorted_appearances_tuples)

sorted_total_mi_tuples = sorted(total_mi.items(), key=lambda x: x[1],reverse=True) 
avg_mi = dict(sorted_total_mi_tuples)

print(bar)

num_features = 0
for feature in avg_mi.keys():
    num_features += 1
    avg_mi[feature] = avg_mi[feature] / NUM_RECORDS
    if num_features < 10:
        print(f"{feature:30} had avg MI of {round(avg_mi[feature], 3)}")
    else:
        break

print(bar)
print()

sns.set(style="whitegrid")

# Prepare the data
freq_x = [x for x in sorted_appearances.keys()]
freq_y = [y for y in sorted_appearances.values()]

# Create the first plot
plt.figure(figsize=(10, 6))
sns.barplot(x=freq_x[:MI_GRAPHS_NUM_COLUMNS], y=freq_y[:MI_GRAPHS_NUM_COLUMNS], color='cornflowerblue')
plt.title("Frequency of Feature Appearance after MI Filtering")
plt.xlabel("Feature in Adjudication Data")
plt.ylabel("Frequency of Feature")
plt.xticks(fontsize = 6, rotation = 45)
plt.tight_layout()
plt.savefig("MI_highest_features.png")
plt.clf()

# Create the second plot
freq_y = [avg_mi[y]for y in sorted_appearances.keys()]

plt.figure(figsize=(10, 6))
sns.barplot(x=freq_x[:MI_GRAPHS_NUM_COLUMNS], y=freq_y[:MI_GRAPHS_NUM_COLUMNS], color='skyblue')
plt.title("Average MI Value")
plt.xlabel("Feature in Adjudication Data")
plt.ylabel("MI Value")
plt.xticks(fontsize = 6, rotation = 45)
plt.tight_layout()
plt.savefig("MI_avg_values.png")


