import pandas as pd

data = pd.read_csv('Churn_Modelling.csv')

data.head()

X = data.drop(['RowNumber', 'Exited', 'Surname', 'CustomerId'], axis = 1)
y = data['Exited']

X.head()

categorical_cols = X.select_dtypes(include = ['object']).columns

numerical_cols = X.select_dtypes(include = ['int64', 'float64']).columns

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, make_scorer, roc_curve
from imblearn.over_sampling import SMOTE

for col in numerical_cols:
  plt.figure(figsize = (5, 5))
  sns.histplot(data[col], kde = True, edgecolor = 'black')
  plt.title(f'Distribution of {col}')
  plt.show()

preprocessor = ColumnTransformer(
    transformers = [
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first'), categorical_cols)
    ])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_transformed, y_train)

svm_model = SVC(probability=True, random_state=42)
svm_model.fit(X_resampled, y_resampled)

train_predictions = svm_model.predict(X_resampled)

train_accuracy = accuracy_score(y_resampled, train_predictions)
print("Training Accuracy:", train_accuracy)

y_pred = svm_model.predict(X_test_transformed)

accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
roc_auc = roc_auc_score(y_test, svm_model.predict_proba(X_test_transformed)[:, 1])

print("Accuracy:", accuracy)
print("Classification Report:\n", classification_rep)
print("ROC-AUC Score:", roc_auc)

