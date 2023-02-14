
import json
import pickle
import os, sys
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
#from sklearn.linear_model import LogisticRegression
try:
       from prophet.plot import plot_plotly, plot_components_plotly
       from prophet import Prophet
except:
       !pip install prophet
       from prophet.plot import plot_plotly, plot_components_plotly
       from prophet import Prophet

fileName = "866420b8578e449b8b4546f2ee5adcf8.csv"

hours=['01h', '02h', '03h', '04h', '05h', '06h', '07h', '08h',
       '09h', '10h', '11h', '12h', '13h', '14h', '15h', '16h', '17h', '18h',
       '19h', '20h', '21h', '22h', '23h', '24h']

chosen_contaminant = 'PM10'


station_col = 'NOM ESTACIO'
city_col = 'MUNICIPI'
date_col = 'DATA'
p_mean = "p_mean"
p_min = "p_min"
p_max = "p_max"
year = "y"
month = 'm'
dow = 'DOW'
CONTAMINANT = "CONTAMINANT"
UNITATS = 'UNITATS'
altitude = 'ALTITUD'
region_type = 'AREA URBANA'

def sanitize_filename(s: str)->str:
    return "".join(x for x in s if x.isalnum())

def get_input(local=False):
    if local:
        print("Reading local file")
        return fileName
    dids = os.getenv("DIDS", None)
    if not dids:
        print("No DIDs found in environment. Aborting.")
        return
    dids = json.loads(dids)
    for did in dids:
        filename = f"data/inputs/{did}/0"  # 0 for metadata service
        print(f"Reading asset file {filename}.")
        return filename


def run_prophet(local=False):
    filename = get_input(local)
    forecasts = {}
    df = pd.read_csv(filename)
    df[p_mean] = df[hours].apply(np.nanmean, axis=1)
    contaminant_df = df[(df[CONTAMINANT] == chosen_contaminant)]
    unique_stations = contaminant_df[station_col].unique()
    for station in unique_stations:
        df_ = contaminant_df[contaminant_df[station_col] == station].groupby(date_col).mean()
#         print(df_.head())
        if (len(df_.index) < 24):
            print(f"{station} has no data for {chosen_contaminant}")
            continue
        df_.index = pd.to_datetime(df_.index)
    
#         df_month = df_.groupby(pd.PeriodIndex(df[date_col], freq="M"))[p_mean].mean()
        df_month = df_.resample('M').mean()
        df_month[date_col] = df_month.index
#         print(df_month)
        print(f'Train model for all past months for {station}')
        df_t = df_month[[p_mean, date_col]]
        df_t.columns = ['y', 'ds']
#         print(df_t)
        m = Prophet()
        m.fit(df_t)
        future = m.make_future_dataframe(periods=24)
        future.tail()
        sanitized_station = sanitize_filename(station)
        forecasts[sanitized_station] = m.predict(future)
        
        
        save_dir = f"models/{sanitized_station}/monthly" if local else f"/data/outputs/result/{sanitized_station}"
        os.makedirs(save_dir, exist_ok=True)
        
        forecasts[station].to_csv(f"{save_dir}/results.csv", index=True)
        filename = f"{save_dir}/prophet_model_monthly_{sanitized_station}.pickle"
        with open(filename, "wb") as pickle_file:
            print(f"Pickling results in {filename}")
            pickle.dump(forecasts, pickle_file)
        
    return forecasts

if __name__ == "__main__":
    local = len(sys.argv) == 2 and sys.argv[1] == "local"
    run_prophet2(local)


