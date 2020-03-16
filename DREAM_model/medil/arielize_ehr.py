# -*- coding: utf-8 -*-
# Besiyata Dishmaya  בס"ד
#
# arielize_ehr.py: arielize EHR, build models, and make predictions
#
# ARIELize - [A]t [R]egular [I]ntervals [E]stimate [L]ongitunal data
# Copyright (C) 2018-2020 Ariel Yehuda Israel, M.D. Ph.D.
# Any utilization of this code/algorithm must credit the author
#
# ARIELize Electronic Health Records data from OMOP format
# then uses xgboost to build model and make predictions
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, version 3 of the License
#
# This program is distributed in the hope that it will help medical
# research, but WITHOUT ANY WARRANTY; without even the implied
# warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
#
# The terms of the GNU General Public License are available here:
# <https://www.gnu.org/licenses/>.

import pdb
import os
import pandas as pd
import csv
import re
import textwrap
import shutil
import datetime
import numpy as np
import string
import gc
import xgboost as xgb
import pickle
import scipy.stats as scs
from sklearn import metrics
from scipy import interpolate
from binascii import crc32
from functools import lru_cache
  
if os.name == 'nt':
    APP_DIR_NAME = 'c:/datasets/ehr_challenge'
else:
    APP_DIR_NAME = '/'
    
DIR_CONCEPTS = os.path.join(APP_DIR_NAME, 'concept_codes_final')
DIR_TRAIN = os.path.join(APP_DIR_NAME, 'train')
DIR_SCRATCH = os.path.join(APP_DIR_NAME, 'scratch')
DIR_MODEL = os.path.join(APP_DIR_NAME, 'model')
DIR_INFER = os.path.join(APP_DIR_NAME, 'infer')
DIR_OUTPUT = os.path.join(APP_DIR_NAME, 'output')

CHUNK_SIZE = 1000000
CASE_CONTROL_RATIO = 10
RANDOM_STATE = 26
TIMEPOINTS = ['1d','2d','5d','15d','45d','3m','6m','9m','1y','2y','4y']

KEY_BY_ARIEL_TYPE = {
    'measurement':'measurement_arielkey',
    'observation':'observation_arielkey',
    'drug':'drug_exposure_arielkey',
    'person':'person_id',
    'visit':'visit_type_concept_id',
    'condition':'condition_concept_id',
    'procedure':'procedure_concept_id',
    'days':None
}

XGBOOST_PARAMS_DEFAULT = {
    # au pif!
    'objective': 'binary:logistic',
    'max_depth': 2,
    'learning_rate': 0.02,
    'colsample_bytree': 0.8,
    'gamma': 7,
    'min_child_weight': 50,
    'subsample': 0.8,
    'stopping_rounds': 200,
    'seed': 5
}
XGBOOST_N_ESTIMATORS = 3000
XGBOOST_N_ESTIMATORS_LOW = 300

CHUNKSIZE = 1000000
ONE_DAY = np.timedelta64(1, 'D')
FILLIN_DAYS_NEVER = 123456

NB_FOLDS = 3
NB_PARTS_SAVE = 30
NB_PARTS_LOAD_INFER = 10

NB_MEASUREMENTS = {'frequent': 100, 'top_num': 500, 'top_alpha': 100, 'topdiff_num': 500, 'topdiff_alpha': 100}
NB_OBSERVATIONS = {'frequent':  20, 'top_num': 100, 'top_alpha': 100, 'topdiff_num': 200, 'topdiff_alpha': 100}
NB_TOP_VISIT_TYPES = 500
NB_TOP_CONDITIONS = 500
NB_TOP_PROCEDURES = 250
NB_TOP_DRUGS = 800

NB_EVAL_XGB_STEPS = 8

pd.set_option('use_inf_as_na', True)

KEEP_COLUMNS = {
'condition_occurrence': ['provider_id','condition_start_datetime','condition_end_datetime','person_id','condition_status_concept_id','condition_concept_id','visit_occurrence_id'],
'death': ['person_id','death_datetime','cause_source_concept_id'],
'drug_exposure': ['drug_exposure_arielkey','drug_concept_id','route_concept_id','person_id','refills','quantity','days_supply','provider_id','drug_exposure_id','drug_exposure_start_datetime','drug_exposure_end_datetime','verbatim_end_date','visit_occurrence_id'],
'measurement': ['measurement_arielkey','measurement_concept_id','measurement_type_concept_id','unit_concept_id','measurement_id','measurement_datetime','person_id','value_as_number','value_as_concept_id'],
'observation': ['observation_type_concept_id','provider_id','value_as_concept_id','person_id','value_as_number','observation_concept_id','observation_id','unit_concept_id','observation_datetime','qualifier_concept_id'],
'observation_period': ['person_id','observation_period_id','observation_period_start_date','observation_period_end_date'],
'person': ['year_of_birth','ethnicity_concept_id','provider_id','person_id','month_of_birth','care_site_id','day_of_birth','location_id','race_concept_id','gender_concept_id','birth_datetime'],
'procedure_occurrence': ['procedure_datetime','provider_id','quantity','person_id','procedure_type_concept_id','procedure_source_concept_id','modifier_concept_id','procedure_concept_id','procedure_occurrence_id','visit_occurrence_id'],
'visit_occurrence': ['provider_id','person_id','care_site_id','visit_occurrence_id','visit_start_datetime','visit_end_datetime','visit_concept_id','visit_type_concept_id','admitted_from_concept_id','discharge_to_concept_id','preceding_visit_occurrence_id']
}

KEYS_TABLES = {
    'measurement': ['measurement_concept_id','measurement_type_concept_id', 'unit_concept_id'],
    'observation': ['observation_concept_id','observation_type_concept_id', 'unit_concept_id'],
    'drug_exposure': ['drug_concept_id','route_concept_id']
}

print_w_timestamp_wrapper=textwrap.TextWrapper(subsequent_indent=' '*3, width=shutil.get_terminal_size((80, 20)).columns-1)


def print_w_timestamp(str, wrapper=print_w_timestamp_wrapper):
    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S") + " " + wrapper.fill(str))


def period_string_to_days(period):
    cnt = float(period[:-1])
    unit = period[-1]
    if unit=='d':
        return cnt
    elif unit=='m':
        return int(cnt * 365.25/12)
    elif unit=='y':
        return int(cnt * 365.25)
    else:
        raise Exception('unrecognized unit')

        
def table_pickle_filename(table_name, type_data = None):
    # type_data can be train or infer
    if type_data is None:
        dest_dir = DIR_SCRATCH
    else:
        dest_dir = os.path.join(DIR_SCRATCH, type_data)
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)    
    return os.path.join(dest_dir, f'{table_name}.pkl')


def import_table(csv_file, table_name, restrict_key=None, restrict_values=None):
    with open(csv_file, newline='') as f:
      csv_reader = csv.reader(f)
      csv_headings = next(csv_reader)
    if table_name not in KEYS_TABLES:
        cols_key = None
    else:
        cols_key = KEYS_TABLES[table_name]
    dtypes = {}
    timestamp_cols = []
    usecols = []
    for col in csv_headings:
        if col not in KEEP_COLUMNS[table_name]:
            continue
        if col.endswith('_string') or col.endswith('_source_value') or col.endswith('_reason') or col.endswith('_name') or (col in ['vocabulary_id','table']):
            # ignore string columns
            col_type = str
        elif col.endswith('_id'):
            col_type = float
            usecols.append(col)
        elif col.endswith('date') or col.endswith('time'):
            col_type = str
            timestamp_cols.append(col)
            usecols.append(col)
        else:
            col_type = float
            usecols.append(col)
        dtypes[col] = col_type
    for col in csv_headings:
        if col in usecols:
            if col.endswith('_time') or col.endswith('_date'):
                datetime_col = re.sub('_(?:date|time)', '_datetime', col)
                if (datetime_col != col) and datetime_col in usecols:
                    print_w_timestamp(f'   ignoring column {col}, redundant w/ {datetime_col}')
                    usecols.remove(col)
    print("\n".join([f'  {k:35s}: {v}' for k,v in dtypes.items()]))
    chunk_no = 0
    table_chunks = []
    for df in pd.read_csv(csv_file, dtype=dtypes, usecols=usecols, chunksize = CHUNK_SIZE):
        if restrict_key is not None and restrict_key in df.columns:
            old_len = len(df)
            df = df[df[restrict_key].isin(restrict_values)]
            print_w_timestamp(f"\trestriction by {restrict_key} keeps {len(df):,} / {old_len:,} rows")
        gc.collect()
        for col in timestamp_cols:
            if col in usecols:
                df[col] = pd.to_datetime(df[col])
        gc.collect()
        chunk_no += 1
        print_w_timestamp(f"\tread chunk {chunk_no:2d}")
        table_chunks.append(df.copy())
    df_all = pd.concat(table_chunks)
    print_w_timestamp(f"read {len(df_all):,} rows")
    if cols_key is not None:
        print_w_timestamp(f"\tadding unique key")
        identifier_col = f'{table_name}_arielkey'
        df_key = df_all[cols_key].drop_duplicates()
        df_key.insert(0,identifier_col, range(1,len(df_key)+1))
        df_all = df_key.merge(df_all,on=cols_key)
        for col in cols_key:
            del df_all[col]
    return df_all
            

def save_df(df, table_name, type_data):
    filename = table_pickle_filename(table_name, type_data)
    print_w_timestamp(f"pickling {len(df):,} rows to {filename}")
    df.to_pickle(filename)
    print_w_timestamp(f"done pickling table {table_name}")


def load_df(table_name, type_data=None):
    return pd.read_pickle(table_pickle_filename(table_name, type_data))


def import_files(csv_files, dir, type_data=None, restrict_key=None, restrict_values=None):
    for i, csv_file in enumerate(csv_files):
        table_name = os.path.basename(csv_file).replace('.csv','')
        print_w_timestamp(f"reading file {i+1}/{len(csv_files):,} {csv_file} to {table_name}")
        df = import_table(os.path.join(dir, csv_file), table_name, restrict_key=restrict_key, restrict_values=restrict_values)
        save_df(df, table_name, type_data)

        
def compute_stats(df, table, key, value_col=None):
    df_gb = df.groupby(f'{table}_arielkey')
    df_stats = df_gb.size().to_frame('nb')
    df_stats['nb_persons'] = df_gb.person_id.nunique()
    if value_col is not None:
        df_stats['nb_value'] = df_gb[value_col].count()
    return df_stats
    
    
def match_controls_with_cases(df_cases, df_control_pool, case_control_ratio, patient_id_col, remove_cases_not_matched=False):
    df_controls = pd.DataFrame()
    i = 0
    df_selected_cases = pd.DataFrame()
    df_selected_controls = pd.DataFrame()
    for index_date,cases in df_cases.groupby('index_date'):
        nb_cases = len(cases)
        print_w_timestamp(f"at index_date={index_date}:, found {nb_cases:,} cases")
        control_candidates = df_control_pool[(df_control_pool.index_date==index_date)]
        nb_needed_controls = nb_cases * case_control_ratio
        if len(control_candidates) > 0:
            if remove_cases_not_matched and nb_needed_controls > len(control_candidates):
                nb_selected_cases = len(control_candidates) * case_control_ratio
                selected_cases = cases.head(nb_selected_cases)
            else:
                selected_cases = cases
            nb_taken_controls = min(nb_needed_controls, len(control_candidates))
            selected_controls = control_candidates.sample(n=nb_taken_controls,random_state=RANDOM_STATE+i)
            df_selected_cases = pd.concat([df_selected_cases, selected_cases.copy()],ignore_index=True)
            df_selected_controls = pd.concat([df_selected_controls, selected_controls.copy()], ignore_index=True)
            print_w_timestamp(f"\tadding {len(selected_cases):,} cases, and {len(selected_controls):,} controls")
        i += 1 
    return df_selected_cases, df_selected_controls
    
    
def get_case_controls(do_not_repeat_case_within_days = 15):
    df_visit = load_df('visit_occurrence','train')
    df_visit['visit_date'] = pd.to_datetime(df_visit['visit_start_datetime'].apply(lambda x:x.date()))
    df_visit = df_visit.sort_values(['person_id','visit_date'],ascending=[True,False])
    df_visit = df_visit[['person_id','visit_date']].drop_duplicates()
    df_visit['is_last_visit'] = (df_visit['person_id']!=df_visit['person_id'].shift()) 
    df_death = load_df('death','train')
    df_death['death_date'] = pd.to_datetime(df_death['death_datetime'].apply(lambda x:x.date()))
    df_death = df_death[['person_id','death_date']].drop_duplicates()
    df_visit_death = df_visit.merge(df_death,on=['person_id'],how='left')
    df_visit_death['days_to_death'] = (df_visit_death['death_date'] - df_visit_death['visit_date']).apply(lambda x: x.days)
    df_visit_death['index_date'] =  df_visit_death['visit_date'] + ONE_DAY
    del df_visit_death['visit_date']
    df_cases = df_visit_death[df_visit_death.days_to_death.fillna(10000) <= 180].copy()
    df_cases['death_within_180d'] = True
    df_cases = df_cases.sort_values(['person_id','days_to_death'])
    can_repeat = df_cases.is_last_visit | (df_cases['days_to_death']>df_cases['days_to_death'].shift()+do_not_repeat_case_within_days)
    df_cases = df_cases[can_repeat]
    print_w_timestamp(f"{len(df_cases):,} cases of {df_cases['person_id'].nunique():,} distinct patients")
    
    df_control_pool = df_visit_death[df_visit_death.days_to_death.fillna(10000) > 366].copy() # take a safety margin
    df_control_pool['death_within_180d'] = False
    df_selected_cases, df_selected_controls = match_controls_with_cases(
        df_cases, df_control_pool, case_control_ratio=CASE_CONTROL_RATIO, patient_id_col = 'person_id'
    )
    cols = ['person_id','index_date','days_to_death','death_within_180d','is_last_visit']
    return df_selected_cases[cols], df_selected_controls[cols]


def merge_with_index_dates(type_data, df, column_date, df_index_dates):
    df = df.merge(df_index_dates, on='person_id')
    date_columns = df.select_dtypes(include=['datetime64']).columns
    if type_data == 'train':
        df = df[ df['is_last_visit'] | (df[column_date] <= df['index_date']) ]
        for other_date_column in date_columns:
            if other_date_column not in [column_date,'index_date']:
                date_erase_pos = (~df['is_last_visit']) & (df[other_date_column] > df['index_date'])
                print_w_timestamp(f'erasing date {other_date_column} in {sum(date_erase_pos)} rows')
                df.loc[date_erase_pos,other_date_column] = None
                df[f'days_before_{other_date_column}'] = (df.index_date - df[other_date_column]).apply(lambda x:x.days)
    df['time_before'] = (df['index_date'] - df[column_date])
    df = df.sort_values(['person_id','time_before'])
    df['days_before'] = df['time_before'].apply(lambda x:x.days)
    del df['time_before']
    return df


def build_periods_df(timepoints=TIMEPOINTS):
    periods = []
    last_days = 0
    for timepoint in timepoints:
        new_days = -period_string_to_days(timepoint)
        periods.insert(0,(timepoint,new_days,last_days))
        last_days = new_days-1
    df_ret = pd.DataFrame(periods,columns=['period_name','days_beg','days_end'])
    df_ret['nb_days'] = df_ret['days_end'] - df_ret['days_beg']
    return df_ret

    
def interpolate_values(df, periods_names, periods_days, dateix_col, val_col, ret_type=np.float32):
    if len(df)>1:
        f = interpolate.interp1d(df[dateix_col], df[val_col], kind='linear', bounds_error=False)
        ret = pd.Series(f(periods_days), periods_names)
    else:
        ret = pd.Series(None, periods_names)
    return ret.astype(ret_type)
    

def compute_ariel(df, dateix_col, val_col, groupby_cols, timepoints=TIMEPOINTS, ret_type=np.float32,
    statistics = ['min','max','mean','median','sd','date_min','date_max','dlast_median']):
    gb = df.groupby(groupby_cols)
    periods_days = [period_string_to_days(x) for x in timepoints]
    df_ariel = gb.apply(interpolate_values,periods_names=timepoints,
        periods_days=periods_days,dateix_col=dateix_col,val_col=val_col, ret_type=ret_type)
    df_ariel.insert(0, 'last', gb[val_col].first().astype(ret_type))
    df_diffs = None
    df_diffs = df_ariel.bfill(axis=1).diff(axis=1,periods=-1).iloc[:, :-1]
    df_diffs.columns = ['d' + x for x in df_diffs.columns]
    df_ariel['first'] = gb[val_col].last().astype(ret_type)
    df_ariel['fdate'] = gb[dateix_col].last()
    df_ariel['ldate'] = gb[dateix_col].first()
    for statistic in statistics:
        statistic_fn = {
            'min': pd.core.groupby.GroupBy.min,
            'max': pd.core.groupby.GroupBy.max,
            'mean': pd.core.groupby.GroupBy.mean,
            'median': pd.core.groupby.GroupBy.median,
            'sd': pd.core.groupby.GroupBy.std,
        }
        if statistic in statistic_fn:
            df_ariel[statistic] = statistic_fn.get(statistic)(gb[val_col])
    if 'date_min' in statistics or 'date_max' in statistics:
        df_min_max = df_ariel[['min','max']].reset_index().merge(df[[groupby_cols] + [val_col,dateix_col]])
        if 'date_min' in statistics:
            df_min = df_min_max[df_min_max[val_col]==df_min_max['min']]
            df_min = df_min[~df_min.duplicated([groupby_cols])]
            df_ariel['date_min'] = df_min.set_index(groupby_cols)[dateix_col]
        if 'date_max' in statistics:
            df_max = df_min_max[df_min_max[val_col]==df_min_max['max']]
            df_max = df_max[~df_max.duplicated([groupby_cols])]
            df_ariel['date_max'] = df_max.set_index(groupby_cols)[dateix_col]
        if 'dlast_median' in statistics:
            df_ariel['dlast_median'] = df_ariel['last'] - df_ariel['median']
    for statistic in statistics:
        df_ariel[statistic] = df_ariel[statistic].astype(ret_type)
    if df_diffs is not None:
        df_ret = pd.concat([df_ariel,df_diffs.astype(ret_type)],axis=1)
    else:
        df_ret = df_ariel
    return df_ret


def add_days_before_dates(df_days_before, df, patients_id_col):
    cols = [col for col in df.columns if col.startswith('days_before')]
    if len(cols)>0:
        gb = df.groupby(patients_id_col)
        for col in cols:
            df_days_before[f'{col}_min'] = gb[col].min()
            df_days_before[f'{col}_max'] = gb[col].max()
    return df_days_before


def arielize_measurements_common(
    df_measurement, patients_id_col, index_date_col, key_col, val_col,
    type_ariel,
    measurement_keys,
    timepoints=TIMEPOINTS
    ):
    
    nb_to_process = len(measurement_keys)
    ret_dfs = []
    for i in range(nb_to_process):
        measurement_key = measurement_keys[i]
        measurement_prefix = type_ariel[0].upper()
        measurement_name = f'{measurement_prefix}{measurement_key}'
        
        print_w_timestamp(f"--- querying data for {type_ariel} {i+1}/{nb_to_process}: {measurement_key}")
        df = df_measurement[df_measurement[key_col]==measurement_key]
        if len(df)>0:
            df = df[~df.duplicated([patients_id_col,index_date_col])]
            print_w_timestamp(f"compute ARIEL values for {len(df)} rows")
            df_ariel = compute_ariel(df, dateix_col=index_date_col, val_col=val_col, groupby_cols=patients_id_col, timepoints=timepoints)
            df_ariel.insert(0, key_col, measurement_key)
            ret_dfs.append(df_ariel.copy())
            print_w_timestamp(f"done")
    ret_df = pd.concat(ret_dfs, ignore_index=False)
    return ret_df
        

def do_chi_square_on_counts(df1, df2, col, key='ehr_index'):
    df_counts = pd.concat([
        df1.groupby(col)[key].nunique() / df1[key].nunique(),
        df2.groupby(col)[key].nunique() / df2[key].nunique()], axis=1).fillna(0)
    
    stats = scs.chisquare(df_counts.T)
    df_counts['pval'] = stats.pvalue
    
    return df_counts.sort_values(['pval'])
    
    
def get_most_significant_features(type_data, feature_name, df, max_taken, feature_save_name=None):
    if feature_save_name is None:
        feature_save_name = feature_name
    pkl_name = f'top_{feature_save_name}'
    if type_data == 'train':
        df_dead = df[df.death_within_180d]
        df_alive = df[~df.death_within_180d]
        df_counts = do_chi_square_on_counts(df_dead, df_alive, feature_name)
        df_counts = df_counts.sort_values('pval')
        top_measurements = df_counts.index.to_frame()
        save_df(top_measurements, pkl_name, type_data)
    else:
        top_measurements = load_df(pkl_name, 'train')
    top_measurement_keys = top_measurements.head(max_taken).index.tolist()
    return top_measurement_keys


def get_most_different_features(type_data, feature_name, feature_value, df, max_taken, feature_save_name=None):
    if feature_save_name is None:
        feature_save_name = feature_name
    pkl_name = f'topd_{feature_save_name}'
    if type_data == 'train':
        df = df[pd.notnull(df[feature_value])]
        df_dead = df[df.death_within_180d]
        df_alive = df[~df.death_within_180d]
        df_counts = df_dead[feature_name].value_counts().to_frame('count_in_dead')
        df_counts['count_in_alive'] = df_alive[feature_name].value_counts()
        df_counts = df_counts.fillna(0)
        df_counts['min_count'] = df_counts.min(axis=1)
        df_counts = df_counts[df_counts['min_count']>=2]
        df_counts['pval1'] = np.nan
        df_counts['pval2'] = np.nan
        nb_counts = len(df_counts)
        i = 0
        for feature, min_count in df_counts['min_count'].iteritems():
            i += 1
            if i%100==1:
                print_w_timestamp(f'comparing {i}/{nb_counts} features')
            x = df_dead[df_dead[feature_name]==feature][feature_value]
            y = df_alive[df_alive[feature_name]==feature][feature_value]
            if set(x) == set(y):
                continue
            try:
                df_counts.loc[feature, 'pval1'] = scs.ttest_ind(x, y).pvalue
            except:
                pass
            try:
                df_counts.loc[feature, 'pval2'] = scs.mannwhitneyu(x, y, alternative='two-sided').pvalue
            except:
                pass
        df_counts['pval'] = df_counts[['pval1','pval2']].min(axis=1)
        df_counts = df_counts.sort_values('pval')
        top_measurements = df_counts.index.to_frame()
        save_df(top_measurements, pkl_name, type_data)
    else:
        top_measurements = load_df(pkl_name, 'train')
    top_measurement_keys = top_measurements.head(max_taken).index
    return top_measurement_keys


def arielize_measurements_other(df, patients_id_col, index_date_col, key_col, val_col):
    df = df[~df.duplicated([patients_id_col,key_col])]
    return df.set_index(df[patients_id_col])[[key_col, index_date_col, val_col]]


def arielize_persons(df, patients_id_col):
    df['birth_date'] = pd.to_datetime(df.year_of_birth*10000+df.month_of_birth*100+df.day_of_birth,format='%Y%m%d')
    df['age'] = (df['index_date'] - df['birth_date']).apply(lambda x:x.days) / 365.25
    df_ariel = df[[
        patients_id_col,'year_of_birth','month_of_birth','day_of_birth','age',
        'gender_concept_id','race_concept_id','ethnicity_concept_id',
        'provider_id', 'care_site_id','location_id'
    ]].set_index(patients_id_col)
    return df_ariel


def arielize_visits(
    df, patients_id_col, index_date_col, visit_type='visit_type_concept_id',
    timepoints=TIMEPOINTS
    ):

    df['nb_days'] = np.maximum(0.1,(df['visit_end_datetime'] - df['visit_start_datetime']) / ONE_DAY).fillna(0.1)
            
    provider_key = 'provider_id'
    care_key = 'care_site_id'
    
    periods = build_periods_df(timepoints=timepoints)
    period_days = periods.set_index('period_name').nb_days.to_dict()

    def compute_summary_cols(df_i, groupby_cols):
        gb = df_i.groupby(groupby_cols, sort=False)
        df_ret = gb.visit_occurrence_id.count().to_frame('qty')
        df_ret['nb_days'] = gb.nb_days.sum()
        df_ret['nb_providers'] = gb[provider_key].nunique()
        df_ret['nb_cares'] = gb[care_key].nunique()
        for field in ['provider', 'care','admitted_from_concept_id','discharge_to_concept_id']:
            if field == 'provider':
                field_key = provider_key
            else:
                field_key = care_key
            df_i = df_i.sort_values(groupby_cols+['nb_days'])
            df_first = df_i[~df_i.duplicated(groupby_cols,'last')]
            df_ret[f'{field}_top'] = df_first.set_index(groupby_cols)[field_key]
        return df_ret        
    
    df['period_name'] = pd.cut(-df.days_before, bins=[-1E6]+periods.days_end.tolist(), labels=periods.period_name)
    
    df_all_periods = compute_summary_cols(df,[patients_id_col,visit_type])
    df_all_periods['ldate'] = df.groupby([patients_id_col,visit_type]).days_before.min()
    df_all_periods['fdate'] = df.groupby([patients_id_col,visit_type]).days_before.max()
    
    last_visit_datetime = df.groupby([patients_id_col,visit_type]).visit_start_datetime.max()
    df_all_periods['ldayofweek'] = last_visit_datetime.dt.dayofweek
    df_all_periods['ldayofyear'] = last_visit_datetime.dt.dayofyear
    df_all_periods['lleapyear'] = last_visit_datetime.dt.is_leap_year + 0.0
    df_all_periods['lminute'] = last_visit_datetime.dt.minute
    df_all_periods['lhour'] = last_visit_datetime.dt.hour + df_all_periods['lminute']/60
    
    df_periods = compute_summary_cols(df,[patients_id_col,visit_type,'period_name'])
    df_periods = df_periods.unstack(2)
    sort_order_columns = pd.factorize(df_periods.columns.get_level_values(1))[0] + (df_periods.columns.codes[0]*0.1)
    df_periods.columns = df_periods.columns.get_level_values(1).astype(str) + '_' + df_periods.columns.get_level_values(0)
    sorted_columns = [x for _,x in sorted(zip(sort_order_columns,df_periods.columns))]
    df_ariel = pd.concat([df_all_periods, df_periods[sorted_columns]], axis=1)
    df_ariel = df_ariel.reset_index().set_index(patients_id_col)
    return df_ariel
        

def arielize_conditions(
    df, patients_id_col, index_date_col, condition_key='condition_concept_id',
    timepoints=TIMEPOINTS
    ):

    df['is_stopped'] = pd.notnull(df['condition_end_datetime']) + 0.0

    gb = df.groupby([patients_id_col,condition_key])
    df_ariel = gb.size().to_frame('nb')
    df_ariel['nb_stopped'] = gb.is_stopped.sum()
    df_ariel['nb_providers'] = gb.provider_id.nunique()
    df_ariel['nb_status'] = gb.condition_status_concept_id.nunique()
    df_ariel['nb_visits'] = gb.visit_occurrence_id.nunique()
    df_ariel['first'] = gb.days_before.max()
    df_ariel['last'] = gb.days_before.min()
    df_ariel = df_ariel.reset_index().set_index(patients_id_col)
    return df_ariel
        

def arielize_procedures(
    df, patients_id_col, index_date_col, procedure_key='procedure_concept_id',
    timepoints=TIMEPOINTS
    ):

    df['quantity'] = df['quantity'].fillna(1)
    
    gb = df.groupby([patients_id_col,procedure_key])
    df_ariel = gb.procedure_occurrence_id.count().to_frame('nb')
    df_ariel['qty'] = gb.quantity.sum()
    df_ariel['nb_providers'] = gb.provider_id.nunique()
    df_ariel['nb_visits'] = gb.visit_occurrence_id.nunique()
    df_ariel['first'] = gb.days_before.max()
    df_ariel['last'] = gb.days_before.min()
    df_ariel = df_ariel.reset_index().set_index(patients_id_col)
    return df_ariel
        

def arielize_drugs(
    df, patients_id_col, index_date_col, drug_key='drug_exposure_arielkey',
    timepoints=TIMEPOINTS
    ):
    
    df['quantity'] = df['quantity'].fillna(1)
    df['end_days_before'] = (df['index_date'] - df['drug_exposure_end_datetime']).apply(lambda x:x.days)
    df.loc[df.end_days_before < 0, 'end_days_before'] = np.nan
    df['is_ended'] = pd.notnull(df['end_days_before']) + 0.0
        
    gb = df.groupby([patients_id_col,drug_key])
    df_ariel = gb.drug_exposure_id.count().to_frame('nb')
    df_ariel['qty'] = gb.quantity.sum()
    df_ariel['days_supply'] = gb.days_supply.sum()
    df_ariel['nb_refills'] = gb.refills.sum()
    df_ariel['nb_ended'] = gb.is_ended.sum()
    df_ariel['nb_providers'] = gb.provider_id.count()
    df_ariel['nb_visits'] = gb.visit_occurrence_id.nunique()
    df_ariel['first'] = gb.days_before.max()
    df_ariel['last'] = gb.days_before.min()
    df_ariel = df_ariel.reset_index().set_index(patients_id_col)
    return df_ariel
    
    
def arielize_dataset(type_data, df_index_dates, types_ariel=['measurement', 'observation', 'condition', 'drug', 'procedure', 'visit', 'person']):
    patients_id_col='ehr_index'
    df_days_before = pd.DataFrame()
    for i in range(len(types_ariel)):
        type_ariel = types_ariel[i]
        measure_id_col = KEY_BY_ARIEL_TYPE[type_ariel]
        
        if (type_ariel == 'measurement') or (type_ariel == 'observation'):
            if type_ariel == 'measurement':
                nb_markers = NB_MEASUREMENTS
            else:
                nb_markers = NB_OBSERVATIONS
            df_measurement = load_df(type_ariel, type_data)
            df_measurement = merge_with_index_dates(type_data, df_measurement, f'{type_ariel}_datetime', df_index_dates)
            df_days_before = add_days_before_dates(df_days_before, df_measurement, patients_id_col)
            df_measurement_num = df_measurement[pd.notnull(df_measurement.value_as_number)]
            if type_data == 'train':
                df_measurement_stats = compute_stats(df_measurement_num, table=type_ariel, key=measure_id_col, value_col='value_as_number')
                df_measurement_stats = df_measurement_stats.sort_values('nb_persons',ascending=False)
                save_df(df_measurement_stats, f'top_{type_ariel}_num', type_data)
            else:
                df_measurement_stats = load_df(f'top_{type_ariel}_num', 'train')
            
            top_measurement_keys_num = df_measurement_stats.head(nb_markers['frequent']).index
            df_measurement_common_num =  df_measurement_num[df_measurement_num[measure_id_col].isin(top_measurement_keys_num)]
            if len(df_measurement_common_num)>0:
                df_ariel = arielize_measurements_common(
                    df_measurement_num, patients_id_col=patients_id_col, index_date_col='days_before', type_ariel=type_ariel,
                    key_col=measure_id_col, val_col='value_as_number',
                    measurement_keys=top_measurement_keys_num
                )
            else:
                df_ariel = pd.DataFrame()
            out_table = f'{type_ariel}_common_num_ariel'
            save_df(df_ariel, out_table, type_data)
            
            df_measurement_other_num = df_measurement_num[~df_measurement_num[measure_id_col].isin(top_measurement_keys_num)]
            top_keys_other_num = get_most_significant_features(type_data, measure_id_col, df_measurement_other_num, nb_markers['top_num'], f'{type_ariel}_other_num')
            df_measurement_top_other_num = df_measurement_other_num[df_measurement_other_num[measure_id_col].isin(top_keys_other_num)]
            
            df_ariel = arielize_measurements_other(df_measurement_top_other_num, patients_id_col=patients_id_col, index_date_col='days_before',
                key_col=measure_id_col, val_col='value_as_number')
            out_table = f'{type_ariel}_num_other_ariel'
            save_df(df_ariel, out_table, type_data)

            df_measurement_alpha = df_measurement[pd.notnull(df_measurement.value_as_concept_id)]
            top_keys_alpha = get_most_significant_features(type_data, measure_id_col, df_measurement_alpha, nb_markers['top_alpha'], f'{type_ariel}_alpha')
            df_measurement_top_alpha = df_measurement_alpha[df_measurement_alpha[measure_id_col].isin(top_keys_alpha)]
            
            df_ariel = arielize_measurements_other(df_measurement_top_alpha, patients_id_col=patients_id_col, index_date_col='days_before',
                key_col=measure_id_col, val_col='value_as_concept_id')
            out_table = f'{type_ariel}_alpha_other_ariel'
            save_df(df_ariel, out_table, type_data)
            
            df_measurement_other2_num = df_measurement_other_num[~df_measurement_other_num.isin(top_keys_other_num)]
            top_keys_other2_num = get_most_different_features(type_data, measure_id_col, 'value_as_number',
                df_measurement_other2_num, nb_markers['topdiff_num'], f'{type_ariel}_diff_num')
                
            df_measurement_top_other2_num =  df_measurement_other2_num[df_measurement_other2_num[measure_id_col].isin(top_keys_other2_num)]
            
            df_ariel = arielize_measurements_other(df_measurement_top_other2_num, patients_id_col=patients_id_col, index_date_col='days_before',
                key_col=measure_id_col, val_col='value_as_number')
            out_table = f'{type_ariel}_diff_num_ariel'
            save_df(df_ariel, out_table, type_data)

            df_measurement_other2_alpha = df_measurement_alpha[~df_measurement_alpha[measure_id_col].isin(top_keys_alpha)]
            top_keys_other2_alpha = get_most_different_features(type_data, measure_id_col, 'value_as_concept_id',
                df_measurement_other2_alpha, nb_markers['topdiff_alpha'], f'{type_ariel}_diff_alpha')
            df_measurement_top_other2 =  df_measurement_other2_alpha[df_measurement_other2_alpha[measure_id_col].isin(top_keys_other2_alpha)]
            
            df_ariel = arielize_measurements_other(df_measurement_top_other2, patients_id_col=patients_id_col, index_date_col='days_before',
                key_col=measure_id_col, val_col='value_as_concept_id')
        
            out_table = f'{type_ariel}_fiff_alpha_ariel'
            save_df(df_ariel, out_table, type_data)

        elif type_ariel == 'person':
            df = load_df('person',type_data)
            df = df.merge(df_index_dates, on='person_id')
            df_ariel = arielize_persons(
                df, patients_id_col=patients_id_col
            )
            out_table = f'{type_ariel}_ariel'
            save_df(df_ariel, out_table, type_data)
        elif type_ariel == 'visit':
            df = load_df('visit_occurrence',type_data)
            df = merge_with_index_dates(type_data, df, 'visit_start_datetime', df_index_dates)
            df_days_before = add_days_before_dates(df_days_before, df, patients_id_col)
            top_keys = get_most_significant_features(type_data, measure_id_col, df, NB_TOP_VISIT_TYPES)
            df.loc[~df[measure_id_col].isin(top_keys), measure_id_col] = -1
    
            df_ariel = arielize_visits(
                df, patients_id_col=patients_id_col, index_date_col='days_before', 
            )
            out_table = f'{type_ariel}_ariel'
            save_df(df_ariel, out_table, type_data)
        elif type_ariel == 'condition':
            df = load_df('condition_occurrence',type_data)
            df = merge_with_index_dates(type_data, df, 'condition_start_datetime', df_index_dates)
            df_days_before = add_days_before_dates(df_days_before, df, patients_id_col)
            top_keys = get_most_significant_features(type_data, measure_id_col, df, NB_TOP_CONDITIONS)
            df.loc[~df[measure_id_col].isin(top_keys), measure_id_col] = -1
    
            df_ariel = arielize_conditions(
                df, patients_id_col=patients_id_col, index_date_col='days_before', 
            )
            out_table = f'{type_ariel}_ariel'
            save_df(df_ariel, out_table, type_data)
        elif type_ariel == 'procedure':
            df = load_df('procedure_occurrence',type_data)
            df = merge_with_index_dates(type_data, df, 'procedure_datetime', df_index_dates)
            df_days_before = add_days_before_dates(df_days_before, df, patients_id_col)
            top_keys = get_most_significant_features(type_data, measure_id_col, df, NB_TOP_PROCEDURES)
            df.loc[~df[measure_id_col].isin(top_keys), measure_id_col] = -1
    
            df_ariel = arielize_procedures(
                df, patients_id_col=patients_id_col, index_date_col='days_before', 
            )
            out_table = f'{type_ariel}_ariel'
            save_df(df_ariel, out_table, type_data)
        elif type_ariel == 'drug':
            df = load_df('drug_exposure',type_data)
            df = merge_with_index_dates(type_data, df, 'drug_exposure_start_datetime', df_index_dates)
            df_days_before = add_days_before_dates(df_days_before, df, patients_id_col)
            top_keys = get_most_significant_features(type_data, measure_id_col, df, NB_TOP_DRUGS)
            df.loc[~df[measure_id_col].isin(top_keys), measure_id_col] = -1

            df_ariel = arielize_drugs(
                df, patients_id_col=patients_id_col, index_date_col='days_before', 
            )
            out_table = f'{type_ariel}_ariel'
            save_df(df_ariel, out_table, type_data)
        else:
            raise Exception(f"type unimplemented {type_ariel}")
        print_w_timestamp (f"SUCCESSFULLY arielized {type_ariel}")
    save_df(df_days_before, f'days_before_ariel', type_data)


def load_ariel(type_data, df, index, measure_id_col, prefix=''):
    df_ret = pd.DataFrame(index=index)
    if len(df)==0:
        return pd.DataFrame()
    df = df[df.index.isin(index)]
    measures = df[measure_id_col].drop_duplicates().to_list()
    to_merge = []
    for i, measure in enumerate(measures):
        measure_name = f"{prefix}{measure}"
        df_data = df[df[measure_id_col] == measure].copy()
        if len(df_data) > 0:
            del df_data[measure_id_col]
            df_data.columns = [f"{measure_name}.{x}" for x in df_data.columns]
            remove_columns_empty = []
            remove_columns_noninformative = []
            for col in df_data.columns:
                if pd.notnull(df_data[col]).sum() == 0:
                    remove_columns_empty.append(col)
                elif df_data[col].nunique() <= 1 and type_data=='train':
                    remove_columns_noninformative.append(col)
            if len(remove_columns_empty) > 0:
                print_w_timestamp(f'removing empty columns {",".join(remove_columns_empty)}')
                for col in remove_columns_empty:
                    del df_data[col]
            if len(remove_columns_noninformative) > 0:
                print_w_timestamp(f'removing noninformative columns {",".join(remove_columns_noninformative)}')
                for col in remove_columns_noninformative:
                    del df_data[col]
            to_merge.append(df_ret.merge(df_data, left_index=True, right_index=True, how='left'))
    if len(to_merge)>0:
        print_w_timestamp(f"merging {len(to_merge)} dataframes")
        df_ret = pd.concat(to_merge,axis=1)
    else:
        df_ret = pd.DataFrame()
    return df_ret

def load_ariel_set(type_data, df_set, index_col, 
      types_ariel = [
          'measurement_common_num',
          'measurement_num_other','measurement_alpha_other',
          'measurement_diff_num','measurement_fiff_alpha',
          'observation_common_num',
          'observation_num_other','observation_alpha_other',
          'observation_diff_num','observation_fiff_alpha',
          'person','visit','condition','procedure','drug','days_before'
      ]
    ):
    df_ds = df_set.set_index(index_col)
    print_w_timestamp(f"loading dataset {len(df_ds):,}")

    for type_ariel in types_ariel:
        ariel_table_name = f'{type_ariel}_ariel'
        type_ariel_suffix = re.sub(f'^[^_]*_','',type_ariel)
        type_ariel_prefix = type_ariel.replace('_' + type_ariel_suffix,'')
        measure_id_col = KEY_BY_ARIEL_TYPE[type_ariel_prefix]
        print_w_timestamp(f"loading ARIEL {ariel_table_name} key:{measure_id_col} ({type_ariel_prefix},{type_ariel_suffix})")
        if type_ariel.startswith('measurement') or type_ariel.startswith('observation'):
            df = load_df(ariel_table_name,type_data)
            prefix = type_ariel[0] + type_ariel_suffix[0]
            df_ariel = load_ariel(type_data, df, df_set.index, measure_id_col=measure_id_col, prefix=prefix)
            date_cols = [x for x in df_ariel.columns if x.endswith('.ldate') or x.endswith('.fdate')]
            df_ariel[date_cols] = df_ariel[date_cols].fillna(FILLIN_DAYS_NEVER)
        elif type_ariel == 'person':
            df_ariel = load_df(ariel_table_name,type_data)
            df_ariel.columns = [f"x.{x}" for x in df_ariel.columns]
        elif type_ariel == 'days_before':
            df_ariel = load_df(ariel_table_name,type_data)
            df_ariel.columns = [f"y.{x}" for x in df_ariel.columns]
        elif type_ariel == 'visit':
            df = load_df(ariel_table_name,type_data)
            df_ariel = load_ariel(type_data, df, df_set.index, measure_id_col=measure_id_col, prefix='v')
            date_cols = [x for x in df_ariel.columns if x.endswith('ldate') or x.endswith('fdate')]
            df_ariel[date_cols] = df_ariel[date_cols].fillna(FILLIN_DAYS_NEVER)
        elif type_ariel == 'condition':
            df = load_df(ariel_table_name,type_data)
            df_ariel = load_ariel(type_data, df, df_set.index, measure_id_col=measure_id_col, prefix='c')
            date_cols = [x for x in df_ariel.columns if x.endswith('first') or x.endswith('last')]
            df_ariel[date_cols] = df_ariel[date_cols].fillna(FILLIN_DAYS_NEVER)
        elif type_ariel == 'procedure':
            df = load_df(ariel_table_name,type_data)
            df_ariel = load_ariel(type_data, df, df_set.index, measure_id_col=measure_id_col, prefix='p')
            date_cols = [x for x in df_ariel.columns if x.endswith('first') or x.endswith('last')]
            df_ariel[date_cols] = df_ariel[date_cols].fillna(FILLIN_DAYS_NEVER)
        elif type_ariel == 'drug':
            df = load_df(ariel_table_name,type_data)
            df_ariel = load_ariel(type_data, df, df_set.index, measure_id_col=measure_id_col, prefix='d')
            date_cols = [x for x in df_ariel.columns if x.endswith('first') or x.endswith('last')]
            df_ariel[date_cols] = df_ariel[date_cols].fillna(FILLIN_DAYS_NEVER)
        else:
            raise Exception(f"type unimplemented {type_ariel}")

        df_ds = df_ds.merge(df_ariel, left_index=True, right_index=True, how='left')
        print_w_timestamp(f"merged {len(df_ds):,} rows")
        del df_ariel
        gc.collect()
    print_w_timestamp(f"done loading data")
    return df_ds


def get_neg_to_pos_ratio(is_positive):
    pos_count = sum(is_positive)
    neg_count = sum(~is_positive)
    neg_to_pos_ratio = neg_count / pos_count
    print_w_timestamp(f"NEG={neg_count:,}, POS={pos_count:,}, neg_to_pos_ratio={neg_to_pos_ratio:.2f}")
    return neg_to_pos_ratio


def load_train_by_folds(only_folds=None):
    print_w_timestamp(f'load_train_by_folds({only_folds})')
    parts_loaded = []
    df_ds_parts = []
    for part_no in range(NB_PARTS_SAVE):
        if only_folds is not None:
            if part_no % NB_FOLDS not in only_folds:
                continue
        parts_loaded.append(str(part_no))
        df_ds_parts.append(load_df(f'ds_arielized_{part_no}', 'train'))
    df_data = pd.concat(df_ds_parts)
    del df_ds_parts
    gc.collect()
    print_w_timestamp(f'loaded {len(df_data):,} rows from {len(parts_loaded)} parts')
    return df_data
    

def build_xgb_models(type_data='train', y_col='death_within_180d',
                   xgb_model_params=XGBOOST_PARAMS_DEFAULT,
                   use_eval_set=False):

    for fold_no in range(NB_FOLDS):
        model_name_d = f"model_fold_{fold_no}"
        picke_filename = os.path.join(DIR_MODEL,f"{model_name_d}.dat")

        print_w_timestamp(f"xgboost model: {model_name_d}")
        if use_eval_set:
            df_data = load_train_by_folds()
        else:
            complementary_folds = [i for i in range(NB_FOLDS) if i != fold_no]
            df_data = load_train_by_folds(complementary_folds)
        
        if len(df_data) > 10000:
            nb_estimators = XGBOOST_N_ESTIMATORS
        else:
            nb_estimators = XGBOOST_N_ESTIMATORS_LOW
            
        print(f"using {nb_estimators} estimators !")

        model_columns = df_data.columns.difference([y_col,'index_date','person_id','days_to_death','is_last_visit']).to_list()

        df_data_fold_no = df_data['person_id'] % NB_FOLDS
        is_fold = (df_data_fold_no == fold_no)
        train_index = ~is_fold
        test_index = is_fold

        scale_pos_weight = get_neg_to_pos_ratio(df_data[y_col] != 0)

        X_train = df_data[train_index][model_columns].astype(np.float32)
        X_valid = df_data[test_index][model_columns].astype(np.float32)
        y_train = df_data[train_index][y_col].apply(lambda x: 0.0+x)
        y_valid = df_data[test_index][y_col].apply(lambda x: 0.0+x)
        del df_data
        gc.collect()  # garbage collect, to free memory for model building

        xgb_model = xgb.XGBClassifier(
            growpolicy='lossguide',
            tree_method='hist',
            objective=xgb_model_params['objective'],
            learning_rate=xgb_model_params['learning_rate'],
            n_estimators=nb_estimators,
            max_depth=xgb_model_params['max_depth'],
            gamma=xgb_model_params['gamma'],
            subsample=xgb_model_params['subsample'],
            colsample_bytree=xgb_model_params['colsample_bytree'],
            min_child_weight=xgb_model_params['min_child_weight'],
            seed=xgb_model_params['seed'],
            scale_pos_weight=scale_pos_weight,
            nthread=-1
        )

        print_w_timestamp(f"fit model {model_name_d}")
        if use_eval_set:
            eval_set = [(X_train, y_train), (X_valid, y_valid)]
            grid_result = xgb_model.fit(
                X_train, y_train,
                eval_set=eval_set,
                eval_metric='auc',
                early_stopping_rounds=xgb_model_params['stopping_rounds'],
                verbose=True)
        else:
            grid_result = xgb_model.fit(
                X_train, y_train,
                eval_metric='auc',
                verbose=True)
        print_w_timestamp("model fitted")
        
        print_w_timestamp("save model as pickle")
        pickle.dump(xgb_model, open(picke_filename, "wb"))
        
        del X_train
        del y_train
        gc.collect()
        df_data = load_train_by_folds([fold_no])
        X_valid = df_data[model_columns].astype(np.float32)
        y_valid = df_data[y_col].apply(lambda x: 0.0+x)
        print(f"computing AUC on {len(X_valid)} rows of validation set")
        y_pred = xgb_model.predict_proba(X_valid)[:,1]
        auc = metrics.roc_auc_score(y_valid, y_pred)
        print(f"========= AUC: {auc:.6f}")

        print_w_timestamp("show features importance")
        imp_vals = pd.DataFrame(xgb_model.feature_importances_)
        imp_vals.columns = ['importance']
        imp_vals['names'] = xgb_model.get_booster().feature_names
        imp_vals = imp_vals[imp_vals.importance > 0]
        imp_vals = imp_vals.sort_values('importance', ascending=False)
        pd.set_option('display.max_rows', None)
        print(imp_vals.head(100))


def add_missing_columns(df_data, model_columns):
    missing_columns = pd.Index(model_columns).difference(df_data.columns)
    nb_missing_columns = len(missing_columns)
    missings = []
    missing_series = pd.Series(np.nan,index=df_data.index)
    print(f'adding {nb_missing_columns} missing columns')
    for i,col in enumerate(missing_columns):
        new_missing_series = missing_series.copy()
        new_missing_series.name = col
        missings.append(new_missing_series)
    if len(missings)>0:
        df_missings = pd.concat(missings, axis=1)
        df_data = df_data.merge(df_missings, left_index=True, right_index=True)
    return df_data


def eval_xgb_model(type_data='train', y_col='death_within_180d', ntree_limits = None):
    statistic = 'roc'
    stats_index = pd.DataFrame()
    
    df_results = load_train_by_folds()[y_col].to_frame()
    df_stats = pd.DataFrame()
 
    for fold_no in range(NB_FOLDS):
        model_name_d = f"model_fold_{fold_no}"
        picke_filename = os.path.join(DIR_MODEL,f"{model_name_d}.dat")

        print_w_timestamp(f"loading model {picke_filename}")
        xgb_model = pickle.load(open(picke_filename, "rb"))
        print_w_timestamp("loaded model")

        if ntree_limits is None:
            nb_estimators = xgb_model.n_estimators
            ntree_step = nb_estimators // NB_EVAL_XGB_STEPS
            ntree_limits = [x for x in range(0, nb_estimators, ntree_step)]
        
        df_data = load_train_by_folds([fold_no])
        model_columns = xgb_model.get_booster().feature_names
        df_data = add_missing_columns(df_data, model_columns)
        
        print_w_timestamp(f"size valid: {len(df_data):,}")
        
        X_index = df_data.index
        X_valid = df_data[model_columns].astype(np.float32)
        y_valid = df_data[y_col].apply(lambda x: 0.0+x)
        del df_data
        gc.collect()

        for ntree in ntree_limits:
            pred_column = f"y_pred_{fold_no}_{ntree}"
            df_results[pred_column] = np.nan
            y_pred = xgb_model.predict_proba(X_valid, ntree_limit=ntree)[:, 1]
            df_results.loc[X_index, pred_column] = y_pred
            #statistic_fold = metrics.roc_auc_score(y_valid, y_pred)
            #df_stats.loc[ntree, f'{statistic}{fold_no}'] = statistic_fold
            print_w_timestamp(f"fold:{fold_no} ntree:{ntree:3d}")# {statistic}={statistic_fold:.6f}")

    for ntree in ntree_limits:
        summary_col = f'y_pred'
        pred_columns = df_results.columns.intersection(
            [f"y_pred_{fold_no}_{ntree}" for fold_no in range(NB_FOLDS)])

        df_results[summary_col] = df_results[pred_columns].mean(axis=1)
        statistic_all = metrics.roc_auc_score(df_results[y_col].apply(lambda x: 0.0 + x), df_results[summary_col])
        print_w_timestamp(f"ntree:{ntree} overall {statistic}: {statistic_all:f}")
        df_stats.loc[ntree, f'{statistic}'] = statistic_all

        df_stats.to_csv(os.path.join(DIR_MODEL,"eval_model.csv"))

    print(df_stats)
    return df_results


def run_xgb_models(df_data):
    df_results = pd.DataFrame(index=df_data.index)
    try:
        df_stats = pd.read_csv(os.path.join(DIR_MODEL,"eval_model.csv"),index_col=0)
        df_stats = df_stats.sort_values('roc',ascending=False)
        ntree = df_stats.index[0]
        print(f'using best nb trees: {ntree}, which gave {df_stats.loc[0]} in training')
    except:
        ntree = 0
        
    for fold_no in range(NB_FOLDS):
        model_name_d = f"model_fold_{fold_no}"
        picke_filename = os.path.join(DIR_MODEL,f"{model_name_d}.dat")

        print_w_timestamp(f"xgboost model: {model_name_d}")
        if not os.path.exists(picke_filename):
            print_w_timestamp(f"did not find model file for {model_name_d}, skipping")
            continue

        print_w_timestamp(f"loading model {picke_filename}")
        xgb_model = pickle.load(open(picke_filename, "rb"))
        print_w_timestamp("loaded model")
        
        #model_features_file = os.path.join(DIR_MODEL,f'{model_name_d}_features.csv')
        #model_columns = pd.read_csv(model_features_file, index_col=0, squeeze=True)
        model_columns = xgb_model.get_booster().feature_names
        df_data = add_missing_columns(df_data, model_columns)

        X_test = df_data[model_columns].astype(np.float32)
        pred_column = f"y_pred_{fold_no}"
        print(f'running prediction with {model_name_d} and ntree_limit={ntree}')
        df_results[pred_column] = xgb_model.predict_proba(X_test, ntree_limit=ntree)[:, 1]

    return df_results.mean(axis=1)


def do_train(recreate_scratch=True):
    type_data = 'train'
    key_to_df = 'ehr_index'
    if recreate_scratch:
        #concept_files = [f for f in os.listdir(DIR_CONCEPTS) if f.endswith('.csv')]
        #import_files(concept_files, DIR_CONCEPTS)
        import_files(['visit_occurrence.csv','death.csv'], dir=DIR_TRAIN, type_data = type_data)
        df_selected_cases, df_selected_controls = get_case_controls()
        df_training_set = pd.concat([df_selected_cases, df_selected_controls], ignore_index=True)
        df_training_set.insert(0, key_to_df, df_training_set.index)
        save_df(df_training_set, 'training_set', type_data)
        train_files = [f for f in os.listdir(DIR_TRAIN) if f.endswith('.csv')]
        train_files.sort()
        print("','".join(train_files))
        import_files(train_files, DIR_TRAIN, type_data = type_data, restrict_key='person_id', restrict_values=df_training_set.person_id)
        df_training_set = load_df('training_set', type_data)
        arielize_dataset(type_data, df_training_set)
        df_ds = load_ariel_set(type_data, df_training_set, index_col='ehr_index')
        del df_ds['index_date']
        gc.collect()
        for part_no in range(NB_PARTS_SAVE):
            df_ds_part = df_ds[df_ds.person_id % NB_PARTS_SAVE == part_no]
            save_df(df_ds_part, f'ds_arielized_{part_no}', type_data)
        del df_ds
        
    build_xgb_models()
    eval_xgb_model()


def do_infer(recreate_scratch=True, evaluate=False):
    type_data = 'infer'
    key_to_df = 'ehr_index'
    if recreate_scratch:
        import_files(['visit_occurrence.csv'], dir=DIR_INFER, type_data = type_data)
        df_visit = load_df('visit_occurrence',type_data)
        df_infer_set = df_visit.groupby(['person_id']).visit_start_datetime.max().to_frame('index_date')
        df_infer_set['index_date'] = pd.to_datetime(df_infer_set['index_date'].apply(lambda x:x.date())) + ONE_DAY
        df_infer_set = df_infer_set.reset_index()
        df_infer_set.insert(0, key_to_df, df_infer_set.index)
        save_df(df_infer_set, 'infer_set', type_data)
        infer_files = [f for f in os.listdir(DIR_INFER) if f.endswith('.csv')]
        infer_files.sort()
        import_files(infer_files, DIR_INFER, type_data = type_data)
        arielize_dataset(type_data, df_infer_set)
        
    df_infer_set = load_df('infer_set', type_data)
    predictions = []
    for part_no in range(NB_PARTS_LOAD_INFER):
        print_w_timestamp(f"{'-'*20} running predictions on part {part_no+1}/{NB_PARTS_LOAD_INFER}")
        df_infer_part = df_infer_set[(df_infer_set.person_id % NB_PARTS_LOAD_INFER) == part_no]
        df_ds_part = load_ariel_set(type_data, df_infer_part, index_col='ehr_index')
        predictions.append(run_xgb_models(df_ds_part))
    
    df_predictions = pd.concat(predictions).to_frame('score')
    df_ret = df_infer_set[[key_to_df,'person_id']].merge(df_predictions,left_on=key_to_df, right_index=True)
    df_ret = df_ret[["person_id", "score"]]
    df_ret[['person_id','score']].to_csv(os.path.join(DIR_OUTPUT,'predictions.csv'), index = False)
    
    if evaluate:
        print('Evaluating model')
        files = [f[:-4] for f in os.listdir(DIR_INFER) if f.endswith('.csv')]
        if 'death' in files:
            df_death = load_df('death', type_data)
            df_death = df_infer_set.merge(df_death[['person_id','death_datetime']],on='person_id',how='left')
            df_death['outcome'] = (df_death.death_datetime - df_death.index_date) <= (ONE_DAY*180)
            df_ret = df_ret.merge(df_death[['person_id','outcome']])
            roc_auc = metrics.roc_auc_score(df_ret['outcome'], df_ret['score'])
            print(f"ROC AUC: {roc_auc:.5f}")
        else:
            print(f"no death file")
    
    
def list_files():
    train_files = [f[:-4] for f in os.listdir(DIR_TRAIN) if f.endswith('.csv')]
    for file in train_files:
        df = load_df(file,type_data='train')
        print(f"'{file}': [" + ",".join([f"'{x}'" for x in df.columns]) + "]")
        continue
        print(f"{'----' * 4} {file}")
        for col in df.columns:
            print(col, df[col].dtype)
