import pandas as pd
import argparse
from datetime import datetime
import dateutil.relativedelta
import os

class MortalityPrediction:
    def __init__(self, dataFolder, project):
        self.dataFolder = dataFolder
        self.months_cutoff = 6
        self.predictionWindow = 6

        self.endOfData = self.get_end_of_data(dataFolder)
        self.cutoff = self.get_cutoff_date()

        self.data = f"data/{project}/"
        if not os.path.exists(self.data):
            os.mkdir(self.data)

        self.train = f"{self.data}training/"
        if not os.path.exists(self.train):
            os.mkdir(self.train)

        self.eval = f"{self.data}evaluation/"
        if not os.path.exists(self.eval):
            os.mkdir(self.eval)

        self.truePositives = f"{self.data}TP.csv"
        self.trueNegatives = f"{self.data}TN.csv"
        self.required_tables = [
            "condition_occurrence.csv",
            "death.csv",
            "drug_exposure.csv",
            "measurement.csv",
            "observation.csv",
            "observation_period.csv",
            "person.csv",
            "procedure_occurrence.csv",
            "visit_occurrence.csv",
            "measurement.csv"
        ]



    def __repr__(self):
        return "MortalityPrediction()"


    def __str__(self):
        return f"""
Mortality Prediction Object
Location of Data:\t{self.dataFolder}
End of Data:\t\t{self.endOfData}
Cut off date:\t\t{self.cutoff}
Number of months:\t{self.months_cutoff}
True positive:\t{self.truePositives}
True negative:\t{self.trueNegatives}
        """

    def get_end_of_data(self, path):
        death = pd.read_csv(f"{path}/death.csv")
        death["death_date"] = pd.to_datetime(death["death_date"],errors='coerce')
        max_death = max(death["death_date"])
        return max_death


    def get_window_begin(self, months, weeks=0):
        if weeks > 0:
            window_begin = self.cutoff - dateutil.relativedelta.relativedelta(weeks=weeks)
        else:
            window_begin = self.cutoff - dateutil.relativedelta.relativedelta(months=months)
        return window_begin


    def get_cutoff_date(self):
        cutoff = self.endOfData - dateutil.relativedelta.relativedelta(months=self.months_cutoff)
        return cutoff


    def TP_TN_distinction(self):
        if os.path.exists(self.truePositives) and os.path.exists(self.trueNegatives):
            pass
        else:
            death = pd.read_csv(f"{self.dataFolder}/death.csv", usecols=["person_id", "death_date"])
            visits = pd.read_csv(f"{self.dataFolder}/visit_occurrence.csv", usecols=["person_id", "visit_start_date"])
            visits["visit_start_date"] = pd.to_datetime(visits["visit_start_date"],errors='coerce')
            visits = visits[visits["visit_start_date"] <= self.cutoff]
            data = death.merge(visits, on="person_id", how="left")
            data["death_date"] = pd.to_datetime(data["death_date"],errors='coerce')
            data["visit_start_date"] = pd.to_datetime(data["visit_start_date"],errors='coerce')

            data["difference"] = data.apply(lambda x: x["death_date"]-dateutil.relativedelta.relativedelta(months=self.months_cutoff) <= x["visit_start_date"], axis=1)
            TP = data[data["difference"]][["person_id"]]
            TP.drop_duplicates(inplace=True)
            TP.to_csv(self.truePositives, index=False)

            person = pd.read_csv(f"{self.dataFolder}/person.csv")
            TN = person.merge(data, on="person_id", how="left")[["person_id", "difference"]]
            TN.fillna(False, inplace=True)
            TN = TN[~TN["difference"]][["person_id"]]
            TN.drop_duplicates(inplace=True)
            TN.to_csv(self.trueNegatives, index=False)


    def split_data_to_training_evaluation(self, ratio):
        visits = pd.read_csv(f"{self.dataFolder}/visit_occurrence.csv", usecols=["person_id", "visit_start_date"])
        print ("visits loaded", flush=True)
        visits["visit_start_date"] = pd.to_datetime(visits["visit_start_date"],errors='coerce')

        print ("total patients calc", flush=True)
        total_patients = float(len((visits[["person_id"]]).drop_duplicates()))

        i = 0
        week = 0
        eval_ratio = 0
        while (eval_ratio) < ratio:
            print ("=========================", flush=True)
            print ("Splitting", i, flush=True)

            if week > 0 or eval_ratio == -1:
                week += 1
                window_begin = self.get_window_begin(months=0, weeks=week)
            else:
                window_begin = self.get_window_begin(months=i)
            print (f"Window Begin: {window_begin}", flush=True)
            print ("applying cutoff calculation", flush=True)
            evaluation = visits[(visits["visit_start_date"] > window_begin) & (visits["visit_start_date"] <= self.cutoff)][["person_id"]].drop_duplicates()

            print ("calc eval ratio", flush=True)
            eval_ratio = round((float(len(evaluation))/total_patients)*100, 3)

            print ("gathering training", flush=True)
            training = visits[~visits["person_id"].isin(evaluation["person_id"])][["person_id"]].drop_duplicates()

            print ("Pred window size:", i, flush=True)
            #print (round((float(len(training))/total_patients)*100, 3))
            print (f"Evaluation: {eval_ratio}%", flush=True)

            if (eval_ratio-ratio) > 10:
                eval_ratio = -1
            elif eval_ratio == 0.0 and week > 4:
                break
            else:
                i += 1
        return training, evaluation


    def split_tables(self, train, evaluation):
        dates = {
            "condition_occurrence.csv": "condition_start_date",
            "drug_exposure.csv": "drug_exposure_start_date",
            "measurement.csv": "measurement_date",
            "observation.csv":"observation_date",
            "observation_period.csv": "observation_period_start_date",
            "procedure_occurrence.csv": "procedure_date",
            "visit_occurrence.csv": "visit_start_date",
            "measurement.csv": "measurement_date"
        }


        for tab in self.required_tables:
            print (f"splitting {tab}", flush=True)
            for data in pd.read_csv(f"{self.dataFolder}/{tab}", chunksize = 10000000):
                if tab not in ["death.csv", "person.csv"]:
                    data[dates[tab]] = pd.to_datetime(data[dates[tab]],errors='coerce')

                    #TN = pd.read_csv(self.trueNegatives)
                    #TN["TN"] = True
                    print ("Pre filter:", len(data), flush=True)
                    data = data[data[dates[tab]] <= self.cutoff]

                    #filtering = filtering.merge(TN, on="person_id", how="left")
                    #filtering.fillna(False, inplace=True)

                    #print (filtering, flush=True)
                    #filtering = filtering[(filtering["TN"]) & (filtering[dates[tab]] <= self.cutoff)][[f"{tab.split('.')[0]}_id"]]

                    #data = data.merge(filtering, on=f"{tab.split('.')[0]}_id", how="inner")
                    #print ("Post filter:", len(data), flush=True)
                else:
                    pass

                train_data = data.merge(train, on="person_id", how="inner")
                train_data.to_csv(f"{self.train}/{tab}", index=False,mode='a', header=True)
                train_data = None
                print (f"train data for {tab} done")

                if tab != "death.csv":
                    eval_data = data.merge(evaluation, on="person_id", how="inner")
                    eval_data.to_csv(f"{self.eval}/{tab}", index=False,mode='a', header=True)
                    eval_data = None
                else:
                    eval_data = None
                print (f"finished {tab}", flush=True)




    def create_goldstandard(self):
        TP = pd.read_csv(self.truePositives)
        TP["status"] = 1
        evals = pd.read_csv(f"{self.eval}person.csv")

        goldstandard = evals.merge(TP, on="person_id", how="left")[["person_id", "status"]]
        goldstandard.fillna(0, inplace=True)
        goldstandard.to_csv(f"{self.data}goldstandard.csv")


    def prune_visits(self):

        visits = pd.read_csv(f"{self.dataFolder}/visit_occurrence.csv", usecols=["visit_occurrence_id", "person_id", "visit_start_date"])
        visits["visit_start_date"] = pd.to_datetime(visits["visit_start_date"],errors='coerce')
        visits = visits[visits["visit_start_date"] <= self.endOfData]

        visits["cutoff"] = self.cutoff

        TN = pd.read_csv(self.trueNegatives)
        TN["TN"] = True
        visits = visits.merge(TN, on="person_id", how="left")
        visits.fillna(False, inplace=True)

        visits = visits[(visits["TN"]) & (visits["visit_start_date"] <= visits["cutoff"])][["visit_occurrence_id"]]

        return visits


    def check_tables(self, path):
        files = set(os.listdir(path))
        req_tables = set(self.required_tables)
        if len(req_tables.intersection(files)) == len(req_tables):
            return True, None
        else:
            missing_tables = [tab for tab in (req_tables-files)]
            return False, missing_tables


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--datafolder", required=True, help="Path to the folder containing the full OMOP dataset")
    parser.add_argument("-r", "--evalRatio", default=20, help="Percentage of the evaluation dataset to the full dataset")
    parser.add_argument("-p", "--project", required=True, help="Name of the project to store the output")
    args = parser.parse_args()

    mp = MortalityPrediction(args.datafolder, args.project)
    print (mp, flush=True)

    # checks to make sure that all necessary tables are available in the data folder
    status, tables = mp.check_tables(args.datafolder)

    if status:

        # categorize patients as True Positive and True Negative
        mp.TP_TN_distinction()
        print ("True Positive and True Negatives found", flush=True)

        #generate training and evaluation patients
        training, evaluation = mp.split_data_to_training_evaluation(ratio=args.evalRatio)
        print ("training and evaluation done", flush=True)

        # split all tables in the required table list in training and evaluation using the generated patient lists
        mp.split_tables(training, evaluation)
        print ("table split", flush=True)

        # build the goldstandard evaluation benchmark
        mp.create_goldstandard()

    else:
        print ("Tables are missing:", flush=True)
        for tab in tables:
            print (f"\t{tab}", flush=True)
