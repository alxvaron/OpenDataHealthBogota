import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor

# Función: Calcular Precisión
def detectar_imprecisiones(df):
    impreciso_edad = (df['edad'] < 0) | (df['edad'] > 120)
    impreciso_fecha = pd.to_datetime(df['fecha'], errors='coerce').isna()
    impreciso_peso = df['peso'] < 0
    return (impreciso_edad | impreciso_fecha | impreciso_peso).sum(), len(df)

def calcular_precision(df):
    imprecisos, total = detectar_imprecisiones(df)
    return (total - imprecisos) / total

# Función: Calcular Consistencia
def calcular_consistencia(df, columnas_unidades):
    total_columnas = len(columnas_unidades)
    columnas_homogeneas = sum(
        df[col].apply(lambda x: isinstance(x, str) and x.endswith(unidad)).all()
        for col, unidad in columnas_unidades.items() if col in df.columns
    )
    return columnas_homogeneas / total_columnas if total_columnas > 0 else 0

# Función: Calcular coherencia
def calcular_coherencia(df):
    instancias_totales = len(df)
    instancias_repetidas = df.duplicated().sum()
    return 1 - (instancias_repetidas / instancias_totales) if instancias_totales > 0 else 1.0

# Función: Calcular actualidad
def calcular_actualidad(df):
    df['fecha_publicacion'] = pd.to_datetime(df['fecha_publicacion'], errors='coerce')
    df['ultima_modificacion'] = pd.to_datetime(df['ultima_modificacion'], errors='coerce')
    tiempo_actual = datetime.now()
    delta_publicacion = (tiempo_actual - df['fecha_publicacion']).dt.total_seconds() / (24 * 3600)
    delta_modificacion = (tiempo_actual - df['ultima_modificacion']).dt.total_seconds() / (24 * 3600)
    delta_publicacion = delta_publicacion.replace(0, 1e-9)
    return (1 - (delta_modificacion / delta_publicacion)).mean()

# Función: Calcular Volatilidad
def calcular_volatilidad(df):
    df['fecha_publicacion'] = pd.to_datetime(df['fecha_publicacion'], errors='coerce')
    df['fecha_expiracion'] = pd.to_datetime(df['fecha_expiracion'], errors='coerce')
    df['volatilidad'] = (df['fecha_expiracion'] - df['fecha_publicacion']).dt.total_seconds() / (24 * 3600)
    return df['volatilidad'].clip(lower=0).mean()

# Función: Calcular puntualidad
def calcular_puntualidad(df, frescura_ideal):
    df['ultima_modificacion'] = pd.to_datetime(df['ultima_modificacion'], errors='coerce')
    tiempo_actual = datetime.now()
    delta_modificacion = (tiempo_actual - df['ultima_modificacion']).dt.total_seconds() / (24 * 3600)
    puntualidad = 1 - (delta_modificacion / frescura_ideal)
    return puntualidad.clip(lower=0, upper=1).mean()

# Cargar datos
datos_prueba = pd.read_csv('datos_prueba.csv')
datos_bogota = pd.read_csv('datos_bogota.csv')

# Definir columnas con unidades para consistencia
columnas_unidades = {'peso': 'kg', 'altura': 'cm'}

# Calcular métricas
precision_eval = calcular_precision(datos_bogota)
consistencia_eval = calcular_consistencia(datos_bogota, columnas_unidades)
coherencia_eval = calcular_coherencia(datos_bogota)
actualidad_eval = calcular_actualidad(datos_bogota)
volatilidad_eval = calcular_volatilidad(datos_bogota)
puntualidad_eval = calcular_puntualidad(datos_bogota, frescura_ideal=30)

# Preparar datos para el modelo
X_train = datos_prueba.drop(['calidad'], axis=1)
y_train = datos_prueba['calidad']
X_test = datos_bogota.drop(['calidad'], axis=1)
y_test = datos_bogota['calidad']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entrenar y evaluar modelo con Random Forest
modelo = RandomForestRegressor(n_estimators=100, random_state=42)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(modelo, X_train_scaled, y_train, cv=kf, scoring='r2')

print(f'R2 Promedio (Validación Cruzada): {np.mean(cv_scores):.4f}')

modelo.fit(X_train_scaled, y_train)
y_pred = modelo.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error (MSE): {mse:.4f}')
print(f'R2 Score: {r2:.4f}')

# Mostrar resultados finales
metricas = ['precisión', 'consistencia', 'coherencia', 'actualidad', 'volatilidad', 
            'puntualidad', 'completitud', 'validez', 'actualización', 'disponibilidad']

resultados = pd.DataFrame({
    'Métrica': metricas,
    'actualidad': [precision_eval, consistencia_eval, coherencia_eval, actualidad_eval, 
              volatilidad_eval, puntualidad_eval] + list(y_pred[6:])
})
print('\nResultados por Métrica de Calidad:')
print(resultados)
