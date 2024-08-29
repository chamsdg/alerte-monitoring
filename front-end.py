# -*- coding: utf-8 -*-
"""
Created on Mon Aug 26 20:10:41 2024

@author: caidara01
"""


import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go

# Titre de l'application
st.title("Application de Prédiction de Churn")

# Introduction
st.markdown("""
Cette application permet d'évaluer la probabilité qu'un client quitte (ou "churne") notre service, basé sur un modèle de machine learning pré-entrainé.
""")

# URL de l'API FastAPI
api_url = "http://192.168.179.200:8007"

# Récupérer les CustomerIDs depuis l'API
try:
    response = requests.get(f"{api_url}/customer_ids/")
    response.raise_for_status()  # Vérifier si la requête a réussi
    customer_ids = response.json().get("customer_ids", [])
    
    if not customer_ids:
        st.error("Aucun identifiant de client trouvé.")
    else:
        # Sélectionner le CustomerID
        customer_id = st.selectbox("Sélectionner le IdClient:", options=customer_ids)

        # Faire une requête à l'API pour obtenir les informations du client
        if customer_id:
            customer_response = requests.get(f"{api_url}/customer_data/{customer_id}")

            if customer_response.status_code == 200:
                customer_data = customer_response.json()
                #st.write("Données du client (raw) :")
                #st.write(customer_data)  # Afficher les données brutes pour débogage

                if 'error' in customer_data:
                    st.error(customer_data['error'])
                else:
                    # Convertir en DataFrame pour une meilleure présentation
                    customer_df = pd.DataFrame([customer_data])
                    
                    # Afficher les informations du client
                    st.write("Informations du client sélectionné :")
                    st.write(customer_df)

                    # Faire une requête à l'API pour obtenir la prédiction
                    predict_response = requests.get(f"{api_url}/predict/{customer_id}")

                    if predict_response.status_code == 200:
                        result = predict_response.json()
                        probability_of_churn = result["Churn Probability"]

                        # Créer le diagramme circulaire avec une jauge de progression
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=probability_of_churn,
                            title={'text': "Churn Probability"},
                            gauge={
                                'axis': {'range': [0, 100]},
                                'bar': {'color': "red" if probability_of_churn > 50 else "green"},
                                'steps': [
                                    {'range': [0, 50], 'color': "lightgreen"},
                                    {'range': [50, 100], 'color': "lightcoral"}],
                            }
                        ))

                        # Afficher le diagramme
                        st.plotly_chart(fig)

                        # Ajouter un message en fonction de la probabilité de churn avec fond coloré
                        if probability_of_churn > 50:
                            st.markdown(
                                f"<div style='background-color:red; color:white; padding:10px;'>Alert: La probabilité que ce client nous quitte est de : {probability_of_churn:.1f}%</div>",
                                unsafe_allow_html=True
                            )
                        else:
                            st.markdown(
                                f"<div style='background-color:green; color:white; padding:10px;'>Bonne Nouvelle: La probabilité que ce client reste avec nous est de : {probability_of_churn:.1f}%</div>",
                                unsafe_allow_html=True
                            )
                    else:
                        st.error("Erreur lors de la récupération de la prédiction. Veuillez réessayer.")
            else:
                st.error("Erreur lors de la récupération des informations du client. Veuillez réessayer.")
except requests.RequestException as e:
    st.error(f"Erreur de connexion à l'API : {e}")
