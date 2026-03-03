import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from Nettoyage_OK import Cleaner



def Clean_Pipeline(df):
    df2 = df.copy()
    Clean = Cleaner(df=df)
    
    df2 = Clean.separation_name(df2)
    df2 = Clean.suppression_name(df2)
    df2 = Clean.convert_seat(df2)
    df2 = Clean.convert_mileage(df2)
    df2 = Clean.convert_engine(df2)
    df2 = Clean.convert_power(df2)
    df2 = Clean.convert_price(df2)
    df2 = Clean.suppr_new_price(df2)
    df2 = Clean.convert_NaN(df2)
    df2 = Clean.suppr_outlier(df2)
    
    return df2



try:
    df = pd.read_csv(r'train.csv') # Pense à vérifier que tu as le bon chemin !
except:
    print("❌ Pas de fichier CSV trouvé. Vérifie le chemin !")
    raise

print("\n🚿 Lancement de la Pipeline de Nettoyage...")
df2 = Clean_Pipeline(df)

# DÉCOUPAGE
X = df2.drop(columns=['Price_EUR'])
y = df2['Price_EUR']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"✅ Données prêtes : {X_train.shape[0]} lignes pour l'entraînement, {X_test.shape[0]} pour le test.\n")







def Model_Pipeline(X_input):

    numeric_features = X_input.select_dtypes(include=[np.number]).columns.tolist()  
    categorical_features = X_input.select_dtypes(exclude=[np.number]).columns.tolist()  

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),  
        ("scaler", StandardScaler())  
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),  
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first")) # Ajout de drop='first' pour optimiser 
    ])

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),  
            ("cat", categorical_transformer, categorical_features)  
        ]
    )
    return preprocess

preprocess = Model_Pipeline(X_train)







modeles = {
    "Régression Linéaire": Pipeline(steps=[("preprocess", preprocess), ("model", LinearRegression())]),
    "Lasso": Pipeline(steps=[("preprocess", preprocess), ("model", Lasso(alpha=0.1))]),
    "Ridge": Pipeline(steps=[("preprocess", preprocess), ("model", Ridge(alpha=1.0))]),
    "SVR": Pipeline(steps=[("preprocess", preprocess), ("model", SVR(C=1000, epsilon=0.1))]),
    "Random Forest": Pipeline(steps=[("preprocess", preprocess), ("model", RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42))]),
    "XGBoost": Pipeline(steps=[("preprocess", preprocess), ("model", XGBRegressor(n_estimators=50, learning_rate=0.1, max_depth=5, random_state=42))])
}

def afficher_courbe_apprentissage(nom_modele, modele, X, y):
    plt.figure(figsize=(8, 4))
    train_sizes, train_scores, val_scores = learning_curve(
        modele, X, y, cv=3, scoring='r2', n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 5)
    )
    plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', color="#C2185B", linewidth=2, label="Entraînement")
    plt.plot(train_sizes, np.mean(val_scores, axis=1), 'o-', color="#4A148C", linewidth=2, label="Validation")
    plt.fill_between(train_sizes, np.mean(train_scores, axis=1), np.mean(val_scores, axis=1), color="#E1BEE7", alpha=0.3)
    plt.title(f"Courbe : {nom_modele}", fontsize=12, color="#4A148C")
    plt.xlabel("Données utilisées")
    plt.ylabel("Score R2")
    plt.legend(loc="best")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

for nom, algo in modeles.items():
    print(f"\n🔹 Évaluation de : {nom}")
    algo.fit(X_train, y_train)
    predictions = algo.predict(X_test)
    
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    
    print(f"   💰 Erreur (MAE)  : {int(mae)} €")
    print(f"   📏 Punition (RMSE) : {int(rmse)} €")
    print(f"   ⭐ Note (R2)     : {r2:.2%} ")
    
    afficher_courbe_apprentissage(nom, algo, X_train, y_train)








paliers = [100, 500, 1000, 2000, 3000, len(X_train)]

# On habille nos duellistes avec l'usine de prétraitement
svr_duel = Pipeline(steps=[("preprocess", preprocess), ("model", SVR(C=1000, epsilon=0.2))])
xgb_duel = Pipeline(steps=[("preprocess", preprocess), ("model", XGBRegressor(n_estimators=100, random_state=42))])

print(f"\n{'Données':<10} | {'Score SVR':<15} | {'Score XGBoost':<15} | {'Gagnant'}")
print("-" * 65)

for taille in paliers:
    X_sub = X_train[:taille]
    y_sub = y_train[:taille]
    
    svr_duel.fit(X_sub, y_sub)
    score_svr = r2_score(y_test, svr_duel.predict(X_test))
    
    xgb_duel.fit(X_sub, y_sub)
    score_xgb = r2_score(y_test, xgb_duel.predict(X_test))
    
    gagnant = "SVR" if score_svr > score_xgb else "XGBoost"
    print(f"{taille:<10} | {score_svr:.2%}        | {score_xgb:.2%}        | {gagnant}")








param_grid_rf = {'model__n_estimators': [100, 200], 'model__max_depth': [10, None]}
param_grid_xgb = {'model__n_estimators': [100, 200], 'model__learning_rate': [0.05, 0.1], 'model__max_depth': [3, 5]}

resultats_grid = {}
modeles_a_tester = [
    ("Random Forest", Pipeline(steps=[("preprocess", preprocess), ("model", RandomForestRegressor(random_state=42))]), param_grid_rf),
    ("XGBoost", Pipeline(steps=[("preprocess", preprocess), ("model", XGBRegressor(random_state=42))]), param_grid_xgb)
]

for nom, algo, grille in modeles_a_tester:
    print(f"👉 Optimisation de {nom} en cours...")
    grid = GridSearchCV(algo, grille, cv=3, scoring='r2', n_jobs=-1)
    grid.fit(X_train, y_train)
    
    best_pipeline = grid.best_estimator_
    predictions = best_pipeline.predict(X_test)
    
    resultats_grid[nom] = {
        "pipeline": best_pipeline, # On sauvegarde la pipeline entière
        "preds": predictions,
        "r2": r2_score(y_test, predictions),
        "params": grid.best_params_
    }
    print(f"   ✅ Terminé ! Meilleur R2 : {resultats_grid[nom]['r2']:.2%}")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
cols_colors = {"Random Forest": "#E91E63", "XGBoost": "#673AB7"}

for i, (nom, res) in enumerate(resultats_grid.items()):
    sns.scatterplot(x=y_test, y=res["preds"], alpha=0.5, color=cols_colors[nom], ax=axes[i])
    axes[i].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    
    # Nettoyage de l'affichage des paramètres (enlever le "model__")
    param_propres = {k.replace('model__', ''): v for k, v in res['params'].items()}
    axes[i].set_title(f"{nom}\nR2: {res['r2']:.2%} | {param_propres}")
    axes[i].set_xlabel("Prix Réel")
    axes[i].set_ylabel("Prix Prédit")
    axes[i].grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()











best_pipeline = resultats_grid["XGBoost"]["pipeline"]

le_boss = best_pipeline.named_steps["model"]
importances = le_boss.feature_importances_

noms_colonnes = best_pipeline.named_steps["preprocess"].get_feature_names_out()

noms_propres = [nom.split('__')[-1] for nom in noms_colonnes]

df_importances = pd.DataFrame({'Feature': noms_propres, 'Importance': importances})
df_importances = df_importances.sort_values(by='Importance', ascending=False).head(10)

plt.figure(figsize=(12, 6))
sns.barplot(x='Importance', y='Feature', data=df_importances, palette='viridis')
plt.title("Top 10 des critères qui définissent le prix d'une voiture", fontsize=14)
plt.xlabel("Poids de la décision")
plt.ylabel("Critère")
plt.show()

print("\n🎉 FIN DU SCRIPT ! Félicitations, ton pipeline complet est opérationnel. 🚀")







# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import StandardScaler, LabelEncoder
# from sklearn.linear_model import LinearRegression, Ridge, Lasso
# from sklearn.svm import SVR
# from sklearn.ensemble import RandomForestRegressor
# from xgboost import XGBRegressor
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from Nettoyage_OK import Cleaner


# def Clean_Pipeline(df):
#     df2 = df.copy()
#     Clean = Cleaner(df=df)
    
#     df2 = Clean.separation_name(df2)
#     df2 = Clean.suppression_name(df2)
#     df2 = Clean.convert_seat(df2)
#     df2 = Clean.convert_mileage(df2)
#     df2 = Clean.convert_engine(df2)
#     df2 = Clean.convert_power(df2)
#     df2 = Clean.convert_price(df2)
#     df2 = Clean.suppr_new_price(df2)
#     df2 = Clean.convert_NaN(df2)
#     df2 = Clean.suppr_outlier(df2)
    
#     return df2



# try:
#     df = pd.read_csv(r'C:\Users\User\OneDrive\Documents\Github\Projet-Voiture-D-Occasion\Data\train.csv') # Assure-toi que ce chemin est le bon vers ton fichier brut
# except:
#     raise


# df2 = Clean_Pipeline(df)
# print(f"✅ Nettoyage terminé ! Taille du dataset : {df2.shape}")

# for col in df2.select_dtypes(include=['object']).columns:
#     le = LabelEncoder()
#     df2[col] = le.fit_transform(df2[col])

# # DÉCOUPAGE (Une seule fois pour tout le script !)
# X = df2.drop(columns=['Price_EUR'])
# y = df2['Price_EUR']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# print(f"✅ Données prêtes : {X_train.shape[0]} lignes pour l'entraînement, {X_test.shape[0]} pour le test.\n")










# modeles = {
#     "Régression Linéaire": make_pipeline(StandardScaler(), LinearRegression()),
#     "Lasso": make_pipeline(StandardScaler(), Lasso(alpha=0.1)),
#     "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
#     "SVR": make_pipeline(StandardScaler(), SVR(C=1000, epsilon=0.1)),
#     "Random Forest": make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)),
#     "XGBoost": make_pipeline(StandardScaler(), XGBRegressor(n_estimators=50, learning_rate=0.1, max_depth=5, random_state=42))
# }

# def afficher_courbe_apprentissage(nom_modele, modele, X, y):
#     plt.figure(figsize=(8, 4))
#     train_sizes, train_scores, val_scores = learning_curve(
#         modele, X, y, cv=3, scoring='r2', n_jobs=-1, 
#         train_sizes=np.linspace(0.1, 1.0, 5)
#     )
#     plt.plot(train_sizes, np.mean(train_scores, axis=1), 'o-', color="#C2185B", linewidth=2, label="Entraînement")
#     plt.plot(train_sizes, np.mean(val_scores, axis=1), 'o-', color="#4A148C", linewidth=2, label="Validation")
#     plt.fill_between(train_sizes, np.mean(train_scores, axis=1), np.mean(val_scores, axis=1), color="#E1BEE7", alpha=0.3)
#     plt.title(f"Courbe : {nom_modele}", fontsize=12, color="#4A148C")
#     plt.xlabel("Données utilisées")
#     plt.ylabel("Score R2")
#     plt.legend(loc="best")
#     plt.grid(True, linestyle='--', alpha=0.5)
#     plt.show()

# for nom, algo in modeles.items():
#     print(f"\n🔹 Évaluation de : {nom}")
#     algo.fit(X_train, y_train)
#     predictions = algo.predict(X_test)
    
#     mae = mean_absolute_error(y_test, predictions)
#     rmse = np.sqrt(mean_squared_error(y_test, predictions))
#     r2 = r2_score(y_test, predictions)
    
#     print(f"   💰 Erreur (MAE)  : {int(mae)} €")
#     print(f"   📏 Punition (RMSE) : {int(rmse)} €")
#     print(f"   ⭐ Note (R2)     : {r2:.2%} ")
    
#     afficher_courbe_apprentissage(nom, algo, X_train, y_train)












# paliers = [100, 500, 1000, 2000, 3000, len(X_train)]
# svr_duel = make_pipeline(StandardScaler(), SVR(C=1000, epsilon=0.2))
# xgb_duel = XGBRegressor(n_estimators=100, random_state=42)

# print(f"\n{'Données':<10} | {'Score SVR':<15} | {'Score XGBoost':<15} | {'Gagnant'}")
# print("-" * 65)

# for taille in paliers:
#     X_sub = X_train[:taille]
#     y_sub = y_train[:taille]
    
#     svr_duel.fit(X_sub, y_sub)
#     score_svr = r2_score(y_test, svr_duel.predict(X_test))
    
#     xgb_duel.fit(X_sub, y_sub)
#     score_xgb = r2_score(y_test, xgb_duel.predict(X_test))
    
#     gagnant = "SVR" if score_svr > score_xgb else "XGBoost"
#     print(f"{taille:<10} | {score_svr:.2%}        | {score_xgb:.2%}        | {gagnant}")










# param_grid_rf = {'n_estimators': [100, 200], 'max_depth': [10, None]}
# param_grid_xgb = {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1], 'max_depth': [3, 5]}

# resultats_grid = {}
# modeles_a_tester = [
#     ("Random Forest", RandomForestRegressor(random_state=42), param_grid_rf),
#     ("XGBoost", XGBRegressor(random_state=42), param_grid_xgb)
# ]

# for nom, algo, grille in modeles_a_tester:
#     print(f"👉 Optimisation de {nom} en cours...")
#     grid = GridSearchCV(algo, grille, cv=3, scoring='r2', n_jobs=-1)
#     grid.fit(X_train, y_train)
    
#     best_model = grid.best_estimator_
#     predictions = best_model.predict(X_test)
    
#     resultats_grid[nom] = {
#         "model": best_model,
#         "preds": predictions,
#         "r2": r2_score(y_test, predictions),
#         "params": grid.best_params_
#     }
#     print(f"   ✅ Terminé ! Meilleur R2 : {resultats_grid[nom]['r2']:.2%}")

# # Affichage des graphiques finaux
# fig, axes = plt.subplots(1, 2, figsize=(14, 6))
# cols_colors = {"Random Forest": "#E91E63", "XGBoost": "#673AB7"}

# for i, (nom, res) in enumerate(resultats_grid.items()):
#     sns.scatterplot(x=y_test, y=res["preds"], alpha=0.5, color=cols_colors[nom], ax=axes[i])
#     axes[i].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
#     axes[i].set_title(f"{nom}\nR2: {res['r2']:.2%} | {res['params']}")
#     axes[i].set_xlabel("Prix Réel")
#     axes[i].set_ylabel("Prix Prédit")
#     axes[i].grid(True, linestyle='--', alpha=0.5)

# plt.tight_layout()
# plt.show()









# # On récupère le meilleur XGBoost du GridSearch
# le_boss = resultats_grid["XGBoost"]["model"]
# importances = le_boss.feature_importances_

# df_importances = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
# df_importances = df_importances.sort_values(by='Importance', ascending=False).head(10)

# plt.figure(figsize=(10, 5))
# sns.barplot(x='Importance', y='Feature', data=df_importances, palette='viridis')
# plt.title("Top 10 des critères qui définissent le prix d'une voiture", fontsize=14)
# plt.xlabel("Poids de la décision")
# plt.ylabel("Critère")
# plt.show()

# print("\n🎉 FIN DU SCRIPT ! Félicitations, ton pipeline complet est opérationnel. 🚀")























# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.model_selection import learning_curve
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.linear_model import LinearRegression, Ridge, Lasso
# from sklearn.svm import SVR
# from sklearn.ensemble import RandomForestRegressor
# from xgboost import XGBRegressor
# from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder, StandardScaler
# from sklearn.svm import SVR
# from sklearn.ensemble import RandomForestRegressor
# from xgboost import XGBRegressor
# import numpy as np
# import seaborn as sns
# from sklearn.model_selection import GridSearchCV
# import joblib



# try:
#     df2 = pd.read_csv('Data\voitures_ia4.csv')
#     print("Fichier chargé depuis le CSV !")
# except:
#     print("Pas de fichier CSV trouvé, on utilise le df2 déjà en mémoire (si tu l'as).")

# # 2. On définit X (les indices) et y (la cible à deviner)
# # X = Tout le tableau SAUF le prix (car c'est la réponse)
# X = df2.drop(columns=['Price_EUR']) 

# # y = Juste le prix
# y = df2['Price_EUR']

# # 3. LE DÉCOUPAGE (Split)
# # On coupe en deux : 80% pour l'entraînement (Train), 20% pour l'examen (Test)
# # random_state=42 : Pour que le mélange soit toujours le même (reproductible)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# print("-" * 30)
# print(f"✅ C'est prêt !")
# print(f"X_train (Entraînement) : {X_train.shape} lignes")
# print(f"X_test (Test) : {X_test.shape} lignes")



















# # 1. On prépare l'arène
# modeles = {
#     "Régression Linéaire": make_pipeline(StandardScaler(), LinearRegression()),
#     "Lasso": make_pipeline(StandardScaler(), Lasso(alpha=0.1)),
#     "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
#     "SVR": make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2)),
#     "Random Forest": make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)),
#     "XGBoost": make_pipeline(StandardScaler(), XGBRegressor(n_estimators=50, learning_rate=0.1, max_depth=5, random_state=42))
# }

# # 2. La fonction Graphique (inchangée, toujours en rosé)
# def afficher_courbe_apprentissage(nom_modele, modele, X, y):
#     plt.figure(figsize=(10, 5))
#     train_sizes, train_scores, val_scores = learning_curve(
#         modele, X, y, cv=5, scoring='r2', n_jobs=-1, 
#         train_sizes=np.linspace(0.1, 1.0, 5)
#     )
#     train_mean = np.mean(train_scores, axis=1)
#     val_mean = np.mean(val_scores, axis=1)

#     plt.plot(train_sizes, train_mean, 'o-', color="#C2185B", linewidth=2, label="Entraînement")
#     plt.plot(train_sizes, val_mean, 'o-', color="#4A148C", linewidth=2, label="Validation (Réalité)")
#     plt.fill_between(train_sizes, train_mean, val_mean, color="#E1BEE7", alpha=0.3)
    
#     plt.title(f"Courbe : {nom_modele}", fontsize=14, color="#4A148C")
#     plt.xlabel("Données utilisées")
#     plt.ylabel("Score R2")
#     plt.legend(loc="best")
#     plt.grid(color='gainsboro', linestyle='--')
#     plt.show()

# # 3. ACTION ! Le calcul des notes ET des courbes
# print("Correction des copies en cours... 🤓\n")

# for nom, algo in modeles.items():
#     print(f"🔹 {nom}")
    
#     # Étape A : On calcule les NOTES précises (MAE, RMSE, R2)
#     # On entraîne le modèle une fois sur tout le train pour tester sur le test
#     algo.fit(X_train, y_train)
#     predictions = algo.predict(X_test)
    
#     mae = mean_absolute_error(y_test, predictions)
#     rmse = np.sqrt(mean_squared_error(y_test, predictions))
#     r2 = r2_score(y_test, predictions)
    
#     # On affiche le bulletin
#     print(f"   💰 Erreur Moyenne (MAE) : {int(mae)} €  <-- Le plus important !")
#     print(f"   📏 Erreur Punitive (RMSE) : {int(rmse)} €")
#     print(f"   ⭐ Note globale (R2)    : {r2:.2%} ")

#     # Étape B : On affiche le GRAPHIQUE d'évolution
#     afficher_courbe_apprentissage(nom, algo, X_train, y_train)
#     print("-" * 50) # Séparateur
















# # ⚠️ LA CORRECTION EST ICI : On transforme le texte en chiffres
# print("🔧 Traduction des textes (Coimbatore -> 3)...")

# for col in df.select_dtypes(include=['object']).columns:
#     le = LabelEncoder()
#     df[col] = le.fit_transform(df[col])

# # Maintenant "Coimbatore" est devenu un chiffre, le SVR sera content !

# # --- 2. DÉCOUPAGE ---
# X = df.drop(columns=['Price_EUR'])
# y = df['Price_EUR']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # --- 3. LE DUEL SVR vs XGBOOST ---
# print("\n🥊 Début du combat SVR vs XGBoost...")

# # On définit les paliers de données (100 voitures, 500 voitures, etc.)
# paliers = [100, 500, 1000, 2000, 3000, len(X_train)]

# # On prépare les deux concurrents
# svr_model = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
# xgb_model = XGBRegressor(n_estimators=100, random_state=42)

# print(f"\n{'Données':<10} | {'Score SVR':<15} | {'Score XGBoost':<15} | {'Gagnant'}")
# print("-" * 65)

# for taille in paliers:
#     # On prend une petite tranche des données
#     X_sub = X_train[:taille]
#     y_sub = y_train[:taille]
    
#     # 1. Test SVR
#     svr_model.fit(X_sub, y_sub)
#     pred_svr = svr_model.predict(X_test)
#     score_svr = r2_score(y_test, pred_svr)
    
#     # 2. Test XGBoost (pour comparer)
#     xgb_model.fit(X_sub, y_sub)
#     pred_xgb = xgb_model.predict(X_test)
#     score_xgb = r2_score(y_test, pred_xgb)
    
#     # Verdict
#     gagnant = "SVR" if score_svr > score_xgb else "XGBoost"
    
#     print(f"{taille:<10} | {score_svr:.2%}        | {score_xgb:.2%}        | {gagnant}")














# # --- ÉTAPE 3 : Le Découpage (Train / Test) ---
# X = df.drop(columns=['Price_EUR'])
# y = df['Price_EUR']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# print(f"\n3. Découpage terminé : {X_train.shape[0]} voitures pour l'entraînement.")

# # --- ÉTAPE 4 : L'Arène des Combattants ---
# # On va comparer le SVR (ton chouchou du moment) avec les champions (RF et XGB)
# # IMPORTANT : Le SVR a BESOIN du StandardScaler. Les autres s'en foutent, mais ça ne leur fait pas de mal.

# modeles = {
#     "SVR (Support Vector Regressor)": make_pipeline(StandardScaler(), SVR(C=1000, epsilon=0.1)), 
#     # J'ai boosté un peu le C du SVR (C=1000) pour qu'il soit moins timide
    
#     "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
#     "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
# }

# print("\n4. Lancement du combat... 🥊")
# print("-" * 60)
# print(f"{'Modèle':<30} | {'Erreur (MAE)':<15} | {'Score R2':<10}")
# print("-" * 60)

# resultats = {}

# for nom, algo in modeles.items():
#     # Entraînement
#     algo.fit(X_train, y_train)
    
#     # Prédiction
#     predictions = algo.predict(X_test)
    
#     # Notation
#     mae = mean_absolute_error(y_test, predictions)
#     r2 = r2_score(y_test, predictions)
    
#     resultats[nom] = predictions # On garde les prédictions pour le graphique
    
#     # Affichage propre
#     emoj = "😐"
#     if r2 > 0.80: emoj = "🙂"
#     if r2 > 0.90: emoj = "🤩"
    
#     print(f"{nom:<30} | {int(mae)} €          | {r2:.2%} {emoj}")

# print("-" * 60)

# # --- ÉTAPE 5 : Le Graphique de Vérité pour le SVR ---
# # On va dessiner : "Prix Réel" vs "Prix Prédit par SVR"
# plt.figure(figsize=(10, 6))
# sns.scatterplot(x=y_test, y=resultats["SVR (Support Vector Regressor)"], alpha=0.5, color="purple")

# # La ligne rouge parfaite (si tout était parfait, tous les points seraient dessus)
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)

# plt.title("SVR : Réalité vs Prédiction (Plus c'est proche de la ligne rouge, mieux c'est)")
# plt.xlabel("Vrai Prix")
# plt.ylabel("Prix Prédit par SVR")
# plt.show()























# # Découpage
# X = df.drop(columns=['Price_EUR'])
# y = df['Price_EUR']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # --- 2. CONFIGURATION DU GRIDSEARCH (LES RÉGLAGES À TESTER) ---

# # A. Grille pour Random Forest
# param_grid_rf = {
#     'n_estimators': [100, 200],      # Nombre d'arbres
#     'max_depth': [10, 20, None],     # Profondeur des racines
#     'min_samples_split': [2, 5]      # Minimum pour diviser une branche
# }

# # B. Grille pour XGBoost
# param_grid_xgb = {
#     'n_estimators': [100, 200],
#     'learning_rate': [0.05, 0.1],    # Vitesse d'apprentissage
#     'max_depth': [3, 5, 7]           # Profondeur
# }

# # --- 3. L'ENTRAÎNEMENT (LA PARTIE LONGUE) ---
# print("\n🔥 Démarrage du GridSearch... Va chercher un café, ça va chauffer ! ☕")

# # Dictionnaire pour stocker les résultats
# resultats = {}

# modeles_a_tester = [
#     ("Random Forest", RandomForestRegressor(random_state=42), param_grid_rf),
#     ("XGBoost", XGBRegressor(random_state=42), param_grid_xgb)
# ]

# for nom, algo, grille in modeles_a_tester:
#     print(f"\n👉 Optimisation de {nom} en cours...")
    
#     # On lance la recherche
#     grid = GridSearchCV(algo, grille, cv=3, scoring='r2', n_jobs=-1, verbose=1)
#     grid.fit(X_train, y_train)
    
#     # On récupère le CHAMPION
#     best_model = grid.best_estimator_
#     best_params = grid.best_params_
    
#     # On le teste sur les données cachées
#     predictions = best_model.predict(X_test)
    
#     # Calcul des scores demandés
#     mse = mean_squared_error(y_test, predictions)
#     rmse = np.sqrt(mse)
#     r2 = r2_score(y_test, predictions)
#     mae = mean_absolute_error(y_test, predictions)
    
#     # On stocke tout
#     resultats[nom] = {
#         "model": best_model,
#         "preds": predictions,
#         "mse": mse,
#         "rmse": rmse,
#         "r2": r2,
#         "mae": mae,
#         "params": best_params
#     }
    
#     print(f"   ✅ Terminé ! Meilleur R2 : {r2:.2%}")

# # --- 4. AFFICHAGE DES SCORES FINAUX ---
# print("\n" + "="*50)
# print("🏆 RÉSULTATS DU DUEL FINAL")
# print("="*50)

# for nom, res in resultats.items():
#     print(f"\n🔹 {nom.upper()}")
#     print(f"   ⚙️ Meilleurs réglages : {res['params']}")
#     print(f"   ⭐ R2 Score (Précision) : {res['r2']:.2%}  <-- Le plus important")
#     print(f"   💰 RMSE (Erreur Punitive): {int(res['rmse'])} €")
#     print(f"   📉 MSE (Erreur au carré) : {int(res['mse']):,} (Chiffre énorme, c'est normal)")

# # --- 5. LE GRAPHIQUE COMPARATIF ---
# plt.figure(figsize=(14, 6))

# # Graphique 1 : Random Forest
# plt.subplot(1, 2, 1)
# sns.scatterplot(x=y_test, y=resultats["Random Forest"]["preds"], alpha=0.5, color="#E91E63")
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2) # Ligne noire diagonale
# plt.title(f"Random Forest (R2: {resultats['Random Forest']['r2']:.2%})")
# plt.xlabel("Prix Réel")
# plt.ylabel("Prix Prédit")
# plt.grid(True, linestyle='--', alpha=0.5)

# # Graphique 2 : XGBoost
# plt.subplot(1, 2, 2)
# sns.scatterplot(x=y_test, y=resultats["XGBoost"]["preds"], alpha=0.5, color="#673AB7")
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2) # Ligne noire diagonale
# plt.title(f"XGBoost (R2: {resultats['XGBoost']['r2']:.2%})")
# plt.xlabel("Prix Réel")
# plt.ylabel("Prix Prédit")
# plt.grid(True, linestyle='--', alpha=0.5)

# plt.tight_layout()
# plt.show()






























# # Découpage
# X = df.drop(columns=['Price_EUR'])
# y = df['Price_EUR']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # --- 2. CONFIGURATION DU GRIDSEARCH ---
# param_grid_rf = {
#     'n_estimators': [100, 200],
#     'max_depth': [10, 20, None],
#     'min_samples_split': [2, 5]
# }

# param_grid_xgb = {
#     'n_estimators': [100, 200],
#     'learning_rate': [0.05, 0.1],
#     'max_depth': [3, 5, 7]
# }

# # --- 3. L'ENTRAÎNEMENT ET CALCUL DES COURBES ---
# print("\n🔥 Démarrage de l'optimisation... Patience ! ☕")

# resultats = {}

# modeles_a_tester = [
#     ("Random Forest", RandomForestRegressor(random_state=42), param_grid_rf),
#     ("XGBoost", XGBRegressor(random_state=42), param_grid_xgb)
# ]

# for nom, algo, grille in modeles_a_tester:
#     print(f"\n👉 Optimisation de {nom} en cours...")
    
#     # A. GridSearch (Trouver le meilleur modèle)
#     grid = GridSearchCV(algo, grille, cv=3, scoring='r2', n_jobs=-1, verbose=1)
#     grid.fit(X_train, y_train)
    
#     best_model = grid.best_estimator_
    
#     # B. Scores sur le Test
#     predictions = best_model.predict(X_test)
#     r2 = r2_score(y_test, predictions)
#     rmse = np.sqrt(mean_squared_error(y_test, predictions))
#     mae = mean_absolute_error(y_test, predictions)
    
#     # C. Calcul de la Courbe d'Apprentissage (Learning Curve) sur le MEILLEUR modèle
#     print(f"   📊 Calcul de la courbe de progression pour {nom}...")
#     train_sizes, train_scores, val_scores = learning_curve(
#         best_model, X_train, y_train, cv=3, scoring='r2', n_jobs=-1,
#         train_sizes=np.linspace(0.1, 1.0, 5)
#     )
    
#     # Stockage des résultats
#     resultats[nom] = {
#         "model": best_model,
#         "preds": predictions,
#         "r2": r2,
#         "rmse": rmse,
#         "mae": mae,
#         "params": grid.best_params_,
#         "curve_sizes": train_sizes,
#         "curve_train": np.mean(train_scores, axis=1),
#         "curve_val": np.mean(val_scores, axis=1)
#     }

# # --- 4. AFFICHAGE DES RÉSULTATS ---
# print("\n" + "="*60)
# print("🏆 RÉSULTATS FINAUX")
# print("="*60)

# for nom, res in resultats.items():
#     print(f"\n🔹 {nom.upper()} (Meilleurs réglages : {res['params']})")
#     print(f"   💰 Erreur Moyenne (MAE) : {int(res['mae'])} €")
#     print(f"   ⭐ Précision (R2)       : {res['r2']:.2%}")

# # --- 5. LA GRANDE PLANCHE DE GRAPHIQUES ---
# fig, axes = plt.subplots(2, 2, figsize=(16, 12)) # 2 lignes, 2 colonnes
# plt.subplots_adjust(hspace=0.4) # Espace entre les lignes

# # Couleurs
# cols = {"Random Forest": "#E91E63", "XGBoost": "#673AB7"} # Rose et Violet

# for i, (nom, res) in enumerate(resultats.items()):
#     col_code = cols[nom]
    
#     # --- GRAPHIQUE DU HAUT : Réalité vs Prédiction ---
#     ax_scatter = axes[0, i]
#     sns.scatterplot(x=y_test, y=res["preds"], alpha=0.5, color=col_code, ax=ax_scatter)
#     # Ligne diagonale parfaite
#     ax_scatter.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
#     ax_scatter.set_title(f"{nom} : Réalité vs Prédictions\nR2 = {res['r2']:.2%}", fontsize=12, fontweight='bold')
#     ax_scatter.set_xlabel("Prix Réel")
#     ax_scatter.set_ylabel("Prix Prédit")
#     ax_scatter.grid(True, linestyle='--', alpha=0.5)
    
#     # --- GRAPHIQUE DU BAS : Courbe d'apprentissage ---
#     ax_curve = axes[1, i]
#     ax_curve.plot(res["curve_sizes"], res["curve_train"], 'o-', color="#C2185B", label="Entraînement")
#     ax_curve.plot(res["curve_sizes"], res["curve_val"], 'o-', color="#4A148C", linewidth=2, label="Validation (Réalité)")
#     # Zone remplie
#     ax_curve.fill_between(res["curve_sizes"], res["curve_train"], res["curve_val"], color="#E1BEE7", alpha=0.3)
    
#     ax_curve.set_title(f"{nom} : Progression de l'apprentissage", fontsize=12)
#     ax_curve.set_xlabel("Nombre de voitures apprises")
#     ax_curve.set_ylabel("Score R2")
#     ax_curve.legend()
#     ax_curve.grid(True, linestyle='--', alpha=0.5)

# plt.show()


































# # 1. On récupère le champion (XGBoost) stocké dans tes résultats
# best_model = resultats["XGBoost"]["model"]

# # 2. On récupère les noms des colonnes (features)
# feature_names = X.columns

# # 3. On demande au modèle ce qu'il préfère
# importances = best_model.feature_importances_

# # 4. On crée un petit tableau propre pour trier
# df_importances = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
# df_importances = df_importances.sort_values(by='Importance', ascending=False).head(10) # Top 10

# # 5. Le Graphique
# plt.figure(figsize=(12, 6))
# sns.barplot(x='Importance', y='Feature', data=df_importances, palette='viridis')
# plt.title('Ce qui compte le plus pour le prix (Top 10)', fontsize=15)
# plt.xlabel('Poids de la décision (Importance)')
# plt.ylabel('Critère')
# plt.show()