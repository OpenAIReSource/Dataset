import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans, DBSCAN
import xgboost as xgb
import folium
import geopandas as gpd


def display(df):
    print(df.head())
    plt.show()


def load_data(filepath):
    df = pd.read_csv(filepath)
    return df


def exploratory_data_analysis(df):
    # Temporal Analysis
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    plt.figure(figsize=(12,6))
    plt.plot(df.resample('M').mean())
    plt.title('Monthly Average of Fires')
    plt.show()

    # Spatial Analysis
    map = folium.Map(location=[-15.0, -55.0], zoom_start=5)
    for index, row in df.iterrows():
        folium.CircleMarker(location=(row['latitude'], row['longitude']), radius=5, color='red', fill=True).add_to(map)
    return map


def preprocess_data(df):
    # Fill missing values
    df.fillna(method='ffill', inplace=True)
    # Feature Engineering
    df['fire_season'] = df['date'].dt.month.apply(lambda x: 1 if x in [6, 7, 8] else 0)
    return df


def clustering_analysis(df):
    kmeans = KMeans(n_clusters=3)
    df['kmeans_labels'] = kmeans.fit_predict(df[['longitude', 'latitude']])
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    df['dbscan_labels'] = dbscan.fit_predict(df[['longitude', 'latitude']])
    return df


def predictive_modeling(df):
    X = df.drop(['target', 'kmeans_labels', 'dbscan_labels'], axis=1)
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    xgb_model = xgb.XGBClassifier()
    xgb_model.fit(X_train, y_train)
    return rf, xgb_model


def analysis(df):
    rf, xgb_model = predictive_modeling(df)
    # Spatial and Temporal Predictions
    predictions = xgb_model.predict(df[['features']])
    return predictions


# Insights and Recommendations
# This is a placeholder for insights and recommendations based on the analysis done.


def conclusions_and_future_work():
    """
    In this project, we've established a methodology for monitoring and predicting wildfires in the Pantanal region. Future work could explore the integration of more complex models and real-time data ingestion.
    """


df = load_data('bdqueimadas.csv')
eda = exploratory_data_analysis(df)
eda.save('temp_analysis.html')
