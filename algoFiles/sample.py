# First algorithm to predict
import json
import pickle
import os, sys
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression

try:
       from prophet.plot import plot_plotly, plot_components_plotly
       from prophet import Prophet
except:
       !pip install prophet
       from prophet.plot import plot_plotly, plot_components_plotly
       from prophet import Prophet

hours=['01h', '02h', '03h', '04h', '05h', '06h', '07h', '08h',
       '09h', '10h', '11h', '12h', '13h', '14h', '15h', '16h', '17h', '18h',
       '19h', '20h', '21h', '22h', '23h', '24h']
def get_input(local=False):
    if local:
        print("Reading local file")
        return 'ocean.csv'
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
    df = pd.read_csv(filename)
    df_ = df.loc[df.CONTAMINANT == 'PM10'].groupby(['NOM ESTACIO'])
    forecasts = {}
    for station in df_.groups.keys():
        df_0 = df_.get_group(station)[hours+['DATA']]#.mean()
        for hour in hours:
            forecasts[hour] = {}
            print('Train model for:', hour)
            # df_ = stations_grp.get_group(station)[[hour]+['DATA']]
            df_t = df_0[[hour]+['DATA']]
            df_t.columns = ['y', 'ds']
            m = Prophet()
            m.fit(df_t)
            future = m.make_future_dataframe(periods=100)
            future.tail()
            forecasts[hour] = m.predict(future)
    filename = "prophet_model.pickle" if local else "/data/outputs/result"
    with open(filename, "wb") as pickle_file:
        print(f"Pickling results in {filename}")
        pickle.dump(forecasts, pickle_file)


if __name__ == "__main__":
    local = True  
    run_prophet(local)
