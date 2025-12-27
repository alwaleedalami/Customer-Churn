

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, make_scorer, roc_curve, auc
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

data = pd.read_csv('Churn_Modelling.csv')

X = data.drop(columns=['RowNumber', 'CustomerId', 'Surname', 'Exited'])
y = data['Exited']

categorical_cols = X.select_dtypes(include = ['object']).columns
numerical_cols = X.select_dtypes(include = ['int64', 'float64']).columns

for col in numerical_cols:
  plt.figure(figsize = (5, 5))
  sns.histplot(data[col], kde = True, edgecolor = 'black')
  plt.title(f'Distribution of {col}')
  plt.show()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(), categorical_cols)
    ])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_transformed.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train_transformed, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=1
)

train_loss, train_accuracy = model.evaluate(X_train_transformed, y_train, verbose=0)

print(f"Training Loss: {train_loss}")
print(f"Training Accuracy: {train_accuracy}")

test_loss, test_accuracy = model.evaluate(X_test_transformed, y_test, verbose=0)

print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

y_train_pred_prob = model.predict(X_train_transformed)
y_train_pred = (y_train_pred_prob >= 0.5).astype(int)

train_accuracy = accuracy_score(y_train, y_train_pred)
print("Training Accuracy:", train_accuracy)

y_test_pred_prob = model.predict(X_test_transformed)
y_test_pred = (y_test_pred_prob >= 0.5).astype(int)

accuracy = accuracy_score(y_test, y_test_pred)
classification_rep = classification_report(y_test, y_test_pred)

print("Testing Accuracy:", accuracy)
print("Classification Report:\n", classification_rep)

