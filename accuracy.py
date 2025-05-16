# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Importing essential libraries
import numpy as np
import pandas as pd
import pickle
# Import necessary libraries

from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
# Loading the dataset
df = pd.read_csv('NOAA_int.csv',usecols=['Bleaching','Depth','Storms','HumanImpact','Siltation','Dynamite','Poison','Sewage','Industrial','Commercial'])

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)
df = clean_dataset(df)
#import pandas as pd
import matplotlib.pyplot as plt
# read-in data
#data = pd.read_csv('./test.csv', sep='\t') #adjust sep to your needs
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your DataFrame (assuming it's already available as df)
# df = pd.read_csv("your_data.csv")  # Uncomment if loading from a file

# Count occurrences
occurrences = df['Bleaching'].value_counts()

# Define more vibrant colors
colors = ['#3498db', '#e74c3c']  # Blue for Non-Bleaching, Red for Bleaching

# Set Seaborn style for a better look
sns.set_style("whitegrid")

# Create a bar plot with enhanced aesthetics
plt.figure(figsize=(8, 6))
bars = plt.bar(occurrences.index, occurrences.values, color=colors, edgecolor='black', linewidth=1.2)

# Add value labels on top of bars
for bar in bars:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
             f'{int(bar.get_height())}', ha='center', va='bottom', fontsize=12, fontweight='bold')

# Titles and labels
plt.title("Occurrences of Bleaching and Non-Bleaching", fontsize=14, fontweight='bold', color='#333')
plt.xlabel("Bleaching (0 = Non-Bleaching, 1 = Bleaching)", fontsize=12, fontweight='bold', color='#555')
plt.ylabel("Frequency", fontsize=12, fontweight='bold', color='#555')

# Customize axes
plt.xticks([0, 1], ['Non-Bleaching', 'Bleaching'], fontsize=11, fontweight='bold', color='#444')
plt.yticks(fontsize=11, fontweight='bold', color='#444')

# Show the plot
plt.show()



# Replacing the 0 values from by NaN
df_copy = df.copy(deep=True)
df_copy[['Depth','Storms','HumanImpact','Siltation','Dynamite','Poison','Sewage','Industrial','Commercial']] = df_copy[['Depth','Storms','HumanImpact','Siltation','Dynamite','Poison','Sewage','Industrial','Commercial']].replace(0,np.NaN)
# Model Building
from sklearn.model_selection import train_test_split
df.drop(df.columns[np.isnan(df).any()], axis=1)
X = df.drop(columns='Bleaching')
y = df['Bleaching']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40,random_state=42)
from sklearn.neural_network  import MLPClassifier
classifier = MLPClassifier(random_state=42)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
# Creating a pickle file for the classifier
filename = 'prediction-modell.pkl'
pickle.dump(classifier, open(filename, 'wb'))
Prescription = ''
filename = 'prediction-modell.pkl'
classifier = pickle.load(open(filename, 'rb'))
# Generate a classification report
class_report = classification_report(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
print("Classification Report:", class_report)
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='coolwarm', fmt='d')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()
data = np.array([[10.5,2,0,1,1,0,1,1,0]])
my_prediction = classifier.predict(data)
print(my_prediction[0])
if my_prediction == 1:
     print('Bleaching Disease')

else:
     print('Not-Bleaching Disease')




from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

error=mean_absolute_error(y_test, y_pred)
print("Mean absolute error : " + str(error))
error1=mean_squared_error(y_test, y_pred)
print("Mean squared error : " + str(error1))

error2=r2_score(y_test, y_pred)
print("r squared error : " + str(error1))


# Prepare data
X = df.drop(columns='Bleaching')
y = df['Bleaching']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.40, random_state=42)

# Initialize MLPClassifier
classifier = MLPClassifier(random_state=42, max_iter=1, warm_start=True)  # warm_start=True allows partial_fit

# Lists to store metrics for plotting
train_accuracies = []
val_accuracies = []

# Train the model incrementally
for epoch in range(1, 101):  # Train for 100 epochs
    classifier.fit(X_train, y_train)

    # Predict on training and validation sets
    y_train_pred = classifier.predict(X_train)
    y_val_pred = classifier.predict(X_val)

    # Compute accuracies
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)

    # Store metrics
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)

    print(f"Epoch {epoch}: Train Accuracy = {train_acc:.4f}, Validation Accuracy = {val_acc:.4f}")

# Plot the accuracy graphs
plt.figure(figsize=(10, 6))
plt.plot(range(1, 101), train_accuracies, label='Training Accuracy', color='blue')
plt.plot(range(1, 101), val_accuracies, label='Validation Accuracy', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy over Epochs')
plt.legend()
plt.grid(True)
plt.show()

from sklearn.inspection import permutation_importance
import numpy as np

# Ensure the model is trained before analyzing feature importance
# Train the model (already done in your code)
classifier.fit(X_train, y_train)

from sklearn.metrics import roc_curve, roc_auc_score

# Get the predicted probabilities for the positive class (Bleaching)
y_prob = classifier.predict_proba(X_test)[:, 1]

# Compute ROC curve values
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

# Compute the AUC (Area Under the Curve)
auc_score = roc_auc_score(y_test, y_prob)
print(f"AUC Score: {auc_score:.2f}")

# Plot the ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f"ROC Curve (AUC = {auc_score:.2f})")
plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random Guess')
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()


# Perform permutation importance analysis
perm_importance = permutation_importance(classifier, X_test, y_test, n_repeats=10, random_state=42)

# Extract feature importance
feature_importances = perm_importance.importances_mean
features = X.columns  # Feature names

# Create a dataframe for better visualization
importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# Print and visualize feature importance
print("Feature Importance:")
print(importance_df)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis')
plt.xlabel('Mean Decrease in Accuracy')
plt.ylabel('Feature')
plt.title('Feature Importance (Permutation)')
plt.show()
