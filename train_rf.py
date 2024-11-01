import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor

# Cargar datos de entrenamiento desde S3
data = pd.read_csv('s3://<tu-bucket>/datos_prueba.csv')
X = data.drop('calidad', axis=1)
y = data['calidad']

# Entrenar el modelo RandomForest
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Guardar el modelo entrenado
joblib.dump(model, '/opt/ml/model/model.joblib')
