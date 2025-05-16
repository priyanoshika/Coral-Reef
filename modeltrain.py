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




from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

error=mean_absolute_error(y_test, y_pred)
print("Mean absolute error : " + str(error))
error1=mean_squared_error(y_test, y_pred)
print("Mean squared error : " + str(error1))

error2=r2_score(y_test, y_pred)
print("r squared error : " + str(error1))
