
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

def construct_hourly_df(cur_df):
    new_df = pd.DataFrame(columns=[date_col, conc_col])
    for idx,row in cur_df.iterrows():        
        
        for idx_h,hour in enumerate(hours):
            new_row = pd.DataFrame({
                date_col: row[date_col] + timedelta(hours=idx_h + 1),
                conc_col: row[hour]
            },index=[0])
            new_df = pd.concat([new_df,new_row]).reset_index(drop=True)
    return new_df

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


def run_prophet2(local=False):
    filename = get_input(local)
#     df = pd.read_csv(filename)
    df_ = df.loc[df.CONTAMINANT == chosen_contaminant].groupby(station_col)
    forecasts = {}
    contaminant_df = df[(df[CONTAMINANT] == chosen_contaminant)]
    unique_stations = contaminant_df[station_col].unique()
    for station in unique_stations:
        df_ = contaminant_df[contaminant_df[station_col] == station].groupby(date_col).mean().iloc[-365:]
        df_[date_col] =  pd.to_datetime(df_.index, format='%d/%m/%Y', errors='coerce')
#         print(df_.tail())
        new_df = construct_hourly_df(df_)
        print('Train model for:', station)
        # df_ = stations_grp.get_group(station)[[hour]+['DATA']]
#         df_t = df_0[[hour]+[date_col]]
        new_df.columns = [ 'ds','y']
        m = Prophet()
        m.fit(new_df)
        future = m.make_future_dataframe(periods=314)
        future.tail()
        sanitized_station = sanitize_filename(station)
        forecasts[sanitized_station] = m.predict(future)
        
        
        save_dir = f"models/{sanitized_station}/hourly" if local else f"/data/outputs/result/{sanitized_station}"
        os.makedirs(save_dir, exist_ok=True)
        
        forecasts[sanitized_station].to_csv(f"{save_dir}/results.csv", index=True)
        filename = f"{save_dir}/prophet_model_hourly_{sanitized_station}.pickle"
        with open(filename, "wb") as pickle_file:
            print(f"Pickling results in {filename}")
            pickle.dump(forecasts, pickle_file)
        
    return forecasts

if __name__ == "__main__":
    local = len(sys.argv) == 2 and sys.argv[1] == "local"
    run_prophet2(local)


