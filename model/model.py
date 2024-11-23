import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import classification_report, mean_squared_error

# Load dataset
data = pd.read_csv('data.csv')

# Split into features and targets
X = data.drop(['Phobia Type', 'Heart Rate Increase (%)', 'Severity'], axis=1)
y_class = data['Phobia Type']  # Categorical target
y_reg = data['Heart Rate Increase (%)']  # Numerical target
y_ord = data['Severity']  # Ordinal categorical target

# Split dataset
X_train, X_test, y_train_class, y_test_class = train_test_split(X, y_class, test_size=0.2, random_state=42)
y_train_reg, y_test_reg = train_test_split(y_reg, test_size=0.2, random_state=42)
y_train_ord, y_test_ord = train_test_split(y_ord, test_size=0.2, random_state=42)

# Preprocessing
categorical_features = ['Gender', 'Occupation', 'Location', 'Lifestyle Factors', 'Cultural Background']
numerical_features = list(set(X.columns) - set(categorical_features))

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ]
)

# Multi-task Random Forest
rf_classifier = RandomForestClassifier(random_state=42)
rf_regressor = RandomForestRegressor(random_state=42)

# Pipelines
class_pipe = Pipeline(steps=[('preprocessor', preprocessor),
                             ('classifier', rf_classifier)])
reg_pipe = Pipeline(steps=[('preprocessor', preprocessor),
                           ('regressor', rf_regressor)])

# Train models
class_pipe.fit(X_train, y_train_class)
reg_pipe.fit(X_train, y_train_reg)

# Predictions
y_pred_class = class_pipe.predict(X_test)
y_pred_reg = reg_pipe.predict(X_test)

# Evaluation
print("Classification Report:")
print(classification_report(y_test_class, y_pred_class))
print("Regression RMSE:", mean_squared_error(y_test_reg, y_pred_reg, squared=False))
