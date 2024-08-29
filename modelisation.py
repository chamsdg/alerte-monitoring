
# import des librairies
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Charger les données
data = pd.read_csv('Churn_Modelling.csv')

data = data.drop(['RowNumber','Surname','Gender','Geography'], axis=1)
# Sélectionner les caractéristiques et la cible
X = data.drop(['Exited'], axis=1)  # Enlever la colonne cible
y = data['Exited']

# Diviser les données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Standardiser les données
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Entraîner le modèle
model_rf = RandomForestClassifier()
model_rf.fit(X_train, y_train)

# Faire des prédictions
y_pred = model_rf.predict(X_test)

# Calculer l'accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Calculer la matrice de confusion
cm = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix:\n{cm}")

# Enregistrer le modèle
joblib.dump(model_rf, 'churn_model_rf.pkl')

# Enregistrer le scaler
joblib.dump(sc, 'scaler.pkl')

# Enregistrer les colonnes de caractéristiques
joblib.dump(X.columns, 'model_features.pkl')
