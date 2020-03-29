import pandas as pd
import argparse
from datetime import datetime
import dateutil.relativedelta
import os

def split_tables():

    dates = {
        "condition_occurrence.csv": "condition_start_date",
        "drug_exposure.csv": "drug_exposure_start_date",
        "observation.csv":"observation_date",
        "observation_period.csv": "observation_period_start_date",
        "procedure_occurrence.csv": "procedure_date",
        "visit_occurrence.csv": "visit_start_date"
        
    }
    required_tables = [
        "condition_occurrence.csv",
        "drug_exposure.csv",
        "observation.csv",
        "observation_period.csv",
        "procedure_occurrence.csv",
        "visit_occurrence.csv"
    ]
    date_string = "2018-08-24"
    cut_off_date = datetime.strptime(date_string, '%Y-%m-%d')
    print(cut_off_date, flush = True)
    for tab in required_tables:
        print (f"trimming {tab}", flush=True)
        data = pd.read_csv(f"./{tab}")
        data[dates[tab]] = pd.to_datetime(data[dates[tab]],errors='coerce')
        print("previous shape", flush = True)
        print(data.shape,flush = True)
        data = data[data[dates[tab]]<cut_off_date]
        print("after shape", flush = True)
        print(data.shape, flush = True )
        data.to_csv(f"./trimmed/{tab}", index = False)

split_tables()
