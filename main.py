import pandas as pd
from Nettoyage_OK import Cleaner


# ======================================= Chargement CSV =========================================

chemin = "data/train.csv"

df = pd.read_csv(chemin)


# =================== Pipeline de nettoyage de la donné =======================


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

df_clean = Clean_Pipeline(df)

# df_clean.info()


# Porbablement à mettre dans une fonction

# Code Joachim/July

# def Netoyage(df):

    # Ensemble du proceessus

    # return df_clean

# ==============================================================================



# ======================== EDA (éventuellement automatisé) ====================

# Ou rendue graphique dans l'interface Flask

# =============================================================================



# ========================= Feeture engeneering ================================

df_fe = df_clean.copy


# Utilisation de fonction qui modifie et cré des nouvelle Feature

# Pour ça utilisé la class Feature enggenering

# =============================================================================


# ============= Intervention des fonction deséléction du model ===============

# Pour celà utilisaer la class Model Selection 

# Entrainement et Export sous forme de joblib avec évalution des metric.

# ============================================================================


# ============ Flask interface et utilisation du model =====================

# Créer une page avec le model choisie fonctionnel intégré et éventuellement les test des autre model

# Site web et fonctionnalité probabablement Graphique

# ==============================================================================