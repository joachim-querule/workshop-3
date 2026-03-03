import numpy as np                                  # Numerical operations
import pandas as pd                                 # Data manipulation with DataFrame
import matplotlib.pyplot as plt                      # Plotting

from sklearn.datasets import fetch_california_housing  # Public dataset loader
from sklearn.model_selection import train_test_split    # Train/test split
from sklearn.pipeline import Pipeline                  # Pipeline to prevent leakage
from sklearn.compose import ColumnTransformer          # Different preprocessing per column type
from sklearn.preprocessing import StandardScaler       # Feature scaling (important for SVR)
from sklearn.preprocessing import OneHotEncoder        # Convert categorical to one-hot vectors
from sklearn.impute import SimpleImputer              # Fill missing values (imputation)

from sklearn.linear_model import LinearRegression      # Multivariable regression model
from sklearn.svm import SVR                            # Support Vector Regression model

from sklearn.metrics import mean_absolute_error        # MAE metric
from sklearn.metrics import mean_squared_error         # MSE metric (we will take sqrt -> RMSE)
from sklearn.metrics import r2_score                   # R² metric

# ------------------------------
# 1) DATA CLEANING (load + basic checks)
# ------------------------------

# chemin = "data/train.csv"

# df = pd.read_csv(chemin)

class Feature_Engeennering():
    def __init__(self):
        pass

    def Engeneering(self, df_clean):

        # La question est de savoir qu'est ce que 'lon va mettre exactement.

        print("\n[3) FEATURE ENGINEERING]")
        df_fe = df_clean.copy()                               # Copie du dataframe nettoyé

        # Variables créées (ratios)
        df_fe["RoomsPerHouseAge"] = df_fe["AveRooms"] / df_fe["HouseAge"].clip(lower=1)  # Évite la division par 0
        df_fe["BedroomsPerRoom"] = df_fe["AveBedrms"] / df_fe["AveRooms"].clip(lower=1)  # Évite la division par 0
        df_fe["PopulationPerOccupant"] = df_fe["Population"] / df_fe["AveOccup"].clip(lower=1)  # Évite la division par 0

        # Création d’une variable catégorielle (bandes de latitude)
        df_fe["LatBand"] = pd.cut(                            # Discrétise la latitude en intervalles
            df_fe["Latitude"],                                # Colonne à découper
            bins=4,                                           # Nombre d’intervalles
            labels=["B1", "B2", "B3", "B4"]                    # Étiquettes des intervalles
        )

        # Séparation des variables explicatives (X) et de la cible (y)
        X = df_fe.drop(columns=["target"])                    # Uniquement les variables d’entrée
        y = df_fe["target"]                                   # Uniquement la variable cible


        # En gros ici on crée de nouvelles variables pour aider le modèle à mieux apprendre si nécessaire.
        # On y reviendra probablement plus tard selon l’évolution du projet.


class PIPELINE:
    def __init__(self):
        pass

    def Model_Pipeline(self):
        print("\n[4) SÉLECTION DU MODÈLE + PRÉTRAITEMENT]")

        # ============================== Listage des type de Collonne ===========================================

        numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()   # Liste des colonnes numériques
        categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()  # Liste des colonnes catégorielles

        print("Numeric features:", numeric_features)           # Affiche les variables numériques
        print("Categorical features:", categorical_features)   # Affiche les variables catégorielles
        
        # Prétraitement pour les colonnes numériques : imputation par la médiane + mise à l’échelle
        numeric_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),     # Remplace les valeurs numériques manquantes par la médiane (calculée sur le TRAIN uniquement)
            ("scaler", StandardScaler())                       # Standardise les variables (ajusté sur le TRAIN uniquement)
        ])

        # Prétraitement pour les colonnes catégorielles : imputation par la valeur la plus fréquente + one-hot encoding
        categorical_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),  # Remplace les catégories manquantes par la plus fréquente (TRAIN uniquement)
            ("onehot", OneHotEncoder(handle_unknown="ignore"))     # Encode les catégories en one-hot (appris sur le TRAIN uniquement)
        ])

        # Combine le prétraitement numérique et catégoriel dans un seul objet
        preprocess = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),     # Applique le pipeline numérique aux colonnes numériques
                ("cat", categorical_transformer, categorical_features)  # Applique le pipeline catégoriel aux colonnes catégorielles
            ]
        )
        return preprocess


# ------------------------------
# 5) ENTRAÎNEMENT DES MODÈLES
# CRUCIAL pour éviter la fuite de données : on sépare les données AVANT d’entraîner le prétraitement ou le modèle.
# ------------------------------

class Training_Model():
    def __init__(self):
        pass

    # D'abord on split le set de donner avant de l'intégré.

    def split(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(  # Séparation en jeu d’entraînement et de test
            X, y,                                             # Variables d’entrée et variable cible
            test_size=0.2,                                    # 20% pour le test
            random_state=42                                   # Pour la reproductibilité
        )
        return X_train, X_test, y_train, y_test
    
        # return (f"Train size: {X_train.shape}, Test size: {X_test.shape}")  # Affiche les tailles des jeux de données

    # Entraienment Régression Linéraire

    def Model_Regression_linear(self, X_train, y_train, preprocess):
        # A) Modèle de régression linéaire multivariable (baseline)
        linreg_model = Pipeline(steps=[
            ("preprocess", preprocess),                        # Prétraitement (appris uniquement sur le train)
            ("model", LinearRegression())                      # Modèle de régression linéaire
        ])

        linreg_model.fit(X_train, y_train)                     # Entraîne le pipeline uniquement sur le jeu d’entraînement
        print("Linear Regression trained.")                    # Confirme la fin de l’entraînement

    # Entrainement du model SVR

    def Model_SVR(self, X_train, y_train, preprocess):
        # B) Modèle SVR (SANS recherche d’hyperparamètres, SANS validation croisée)
        # NOTE : Ces hyperparamètres sont choisis manuellement à des fins pédagogiques.
        # Tu peux les ajuster si nécessaire.
        svr_model = Pipeline(steps=[
            ("preprocess", preprocess),                        # Même prétraitement (appris uniquement sur le train)
            ("model", SVR(kernel="rbf", C=10, gamma="scale", epsilon=0.1))  # SVR avec paramètres fixés
        ])

        svr_model.fit(X_train, y_train)                        # Entraîne le pipeline SVR uniquement sur le jeu d’entraînement
        print("SVR trained (fixed hyperparameters).")          # Confirme la fin de l’entraînement


class Evalutaion_Model():
    def __init__(self):
        pass

    def Eval(self, model, X_test, y_test, name="model"):

        preds = model.predict(X_test)                      # Prédictions sur les données de test

        mae = mean_absolute_error(y_test, preds)           # Calcule le MAE
        rmse = np.sqrt(mean_squared_error(y_test, preds))  # Calcule le RMSE
        r2 = r2_score(y_test, preds)                       # Calcule le score R²

        print(f"\n--- {name} ---")                         # Affiche le nom du modèle
        print(f"MAE : {mae:.4f}")                          # Affiche le MAE
        print(f"RMSE: {rmse:.4f}")                         # Affiche le RMSE
        print(f"R^2 : {r2:.4f}")                           # Affiche le R²

        return mae, rmse, r2, preds                               # Retourne les métriques

    def Graph_Metrics(self, y_test, preds, name):
            plt.figure()                                       # Nouvelle figure
            plt.scatter(y_test, preds, s=8)                    # Nuage de points réel vs prédit
            plt.title(f"{name}: Réel vs Prédit")               # Titre
            plt.xlabel("Actual")                               # Label axe X (valeurs réelles)
            plt.ylabel("Predicted")                            # Label axe Y (valeurs prédites)
            plt.show()                                         # Affiche le graphique


    # Utilisation

    # Évaluation de la régression linéaire
    # linreg_scores = evaluate_regression(                   # Évalue le modèle de base
    #     linreg_model,                                      # Pipeline
    #     X_test,                                            # Données de test (features)
    #     y_test,                                            # Données de test (cible)
    #     "Linear Regression"                                # Nom du modèle
    # )

    # # Évaluation du SVR
    # svr_scores = evaluate_regression(                      # Évalue le modèle SVR
    #     svr_model,                                         # Pipeline
    #     X_test,                                            # Données de test (features)
    #     y_test,                                            # Données de test (cible)
    #     "SVR (fixed params)"                               # Nom du modèle
    # )

    # # Création d’un tableau comparatif
    # results = pd.DataFrame(                                # Construit un DataFrame des métriques
    #     [linreg_scores, svr_scores],                       # Lignes = modèles
    #     columns=["MAE", "RMSE", "R2"],                     # Noms des métriques
    #     index=["Linear Regression", "SVR (fixed params)"]  # Noms des modèles
    # )

    # print("\nComparison table:\n", results)                 # Affiche le tableau comparatif