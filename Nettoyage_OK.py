import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv('train.csv')

df2 = df.copy()


class Cleaner():
        
        def __init__(self, df):
            print("Cleaner à bien été charger")
            print("Valeurs manquantes", df.isnull().sum())


#**Séparation 'Name'**


        def separation_name(self, df2):
            df2[['Marque', 'Modele']] = df2['Name'].str.split(' ', n=1, expand=True)
            return df2
        

# Suppression 'Name'

        def suppression_name(self, df2):
            df2.drop(columns=['Name'], inplace=True)
            return df2
        
# Convertion des colonnes 'Seats', 'Mileage', 'Engine', 'Power', 'Price' en numérique

        def convert_seat(self, df2):
            df2['Seats'] = df2['Seats'].fillna(0).astype(int)
            return df2
        
#**Convertir 'Mileage'**

        def convert_mileage(self, df2):
            mask_cng = df2['Mileage'].astype(str).str.contains('km/kg')

            df2['Mileage'] = df2['Mileage'].astype(str).str.replace(' km/kg', '').str.replace(' kmpl', '')
            df2['Mileage'] = pd.to_numeric(df2['Mileage'], errors='coerce')

            df2.loc[mask_cng, 'Mileage'] = df2.loc[mask_cng, 'Mileage'] / 1.4
            return df2
        
#Convertir Engine 

        def convert_engine(self, df2):
            df2['Engine'] = df2['Engine'].astype(str)
            df2['Engine'] = df2['Engine'].str.replace(' CC', '', regex=False)
            df2['Engine'] = pd.to_numeric(df2['Engine'], errors='coerce')
            return df2
        
# **Convertir colonne 'Power'**

        def convert_power(self, df2):
             
            df2['Power'] = df2['Power'].astype(str)

            df2['Power'] = df2['Power'].str.replace(' bhp', '', regex=False)
            df2['Power'] = df2['Power'].str.replace('null', '', regex=False) # Sécurité en plus

            df2['Power'] = pd.to_numeric(df2['Power'], errors='coerce')
            return df2
# Convertire Price en euros

        def convert_price(self, df2):
            df2['Price_EUR'] = df2['Price'] * 1100
            df2['Price_EUR'] = df2['Price_EUR'].astype(int)
            df2.drop(columns=['Price'], inplace=True)
            return df2
        
#**Suppression 'New_Price'**

        def suppr_new_price(self, df2):
            df2.drop(columns=['New_Price'], inplace=True)
            return df2
        
# Comble les NaN (dans Power, Engine et Mileage) par des valeurs similaires selon le Modele, la transmission et le type de carburant 

        def convert_NaN(self, df2):
            colonnes_a_reparer = ['Power', 'Engine', 'Mileage']
            for col in colonnes_a_reparer:
                df2[col] = df2.groupby(['Modele', 'Transmission', 'Fuel_Type'])[col].transform(lambda x: x.ffill().bfill())
                df2.dropna(inplace=True)
            return df2
        
        
# **Valeurs aberrantes**

        def suppr_outlier(self, df2):

            colonnes_a_nettoyer = ['Price_EUR', 'Kilometers_Driven', 'Power']

            for col in colonnes_a_nettoyer:
                limite_haute = df2[col].quantile(0.99)
    
                limite_basse = df2[col].quantile(0.01)
    
                df2 = df2[(df2[col] < limite_haute) & (df2[col] > limite_basse)]
            return df2