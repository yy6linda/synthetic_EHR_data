import pandas as pd
import json
import pyodbc
import sqlalchemy


def database_connection():

    creds = json.load(open('sql_credentials.json'))
        
    database = creds['database']
    username = creds['username']
    password = creds['password']

    server = creds['server']
    driver = '{ODBC Driver 13 for SQL Server}'
   	

    connection_string = f'DRIVER={driver};SERVER={server};PORT=1433;DATABASE={database};UID={username};PWD={password}'
        
    conn = pyodbc.connect(connection_string)
    return conn


def query(sql_string):
    df = pd.read_sql(sql_string, con=database_connection())
    return df


def chunked_query(sql_string, chunk_size):
    df = pd.read_sql(sql_string, con=database_connection(), chunksize=chunk_size)
    return df
