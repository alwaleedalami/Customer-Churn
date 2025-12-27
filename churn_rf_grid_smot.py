
import pandas as pd

data = pd.read_csv('Churn_Modelling.csv')

X = data.drop(['CustomerId', 'RowNumber', 'Exited', 'Surname'], axis = 1)
y = data['Exited']

X.head()

y.head()

categorical_cols = X.select_dtypes(include = ['object']).columns

numerical_cols = X.select_dtypes(include = ['int64', 'float64']).columns

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
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

rf_model = RandomForestClassifier()

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

grid_search = GridSearchCV(rf_model, param_grid, refit=True, verbose=2, cv=5)

grid_search.fit(X_resampled, y_resampled)

print(f"Best Parameters: {grid_search.best_params_}")

train_predictions = grid_search.predict(X_resampled)

train_accuracy = accuracy_score(y_resampled, train_predictions)
print("Training Accuracy:", train_accuracy)

y_pred = grid_search.predict(X_test_transformed)

accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)
roc_auc = roc_auc_score(y_test, grid_search.predict_proba(X_test_transformed)[:, 1])

print("Accuracy:", accuracy)
print("Classification Report:\n", classification_rep)
print("ROC-AUC Score:", roc_auc)

