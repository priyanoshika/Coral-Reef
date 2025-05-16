from sklearn.inspection import permutation_importance


# Importing essential libraries
import numpy as np
import pandas as pd

# Import necessary libraries

from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
# Loading the dataset
df = pd.read_csv('NOAA_int - Copy.csv',usecols=['Bleaching','Depth','Storms','HumanImpact','Siltation','Dynamite','Poison','Sewage','Industrial','Commercial'])

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
import seaborn as sns
sns.countplot(df['Bleaching'],label="Count")
plt.show()
# count occurences
occurrences = df.loc[:, 'Bleaching'].value_counts()
# plot histogram
plt.bar(occurrences.keys(), occurrences)
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
# Generate a classification report
class_report = classification_report(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)
print("Classification Report:", class_report)
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot = True, cmap= 'Blues')
plt.ylabel('True')
plt.xlabel('False')
plt.title('Confusion Matrix')
plt.show()
data = np.array([[10.5,2,0,1,1,0,1,1,0]])
my_prediction = classifier.predict(data)
print(my_prediction[0])
if my_prediction == 1:
     print('Bleaching Disease')

else:
     print('Not-Bleaching Disease')






classifier.fit(X_train, y_train)

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