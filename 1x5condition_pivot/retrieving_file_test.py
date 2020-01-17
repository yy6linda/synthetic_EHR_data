import pandas as pd
import database_functions as dbf
import argparse
from datetime import datetime
import dateutil.relativedelta
import os

def get_cutoff_date():
    query = """
        SELECT MAX(death_datetime) as date
        FROM "OMOP"."death"
    """
    output = dbf.query(query)
    date = output["date"][0].date()
    cutoff = date - dateutil.relativedelta.relativedelta(months=6)
    return cutoff

def select_person():
    cutoff_date = get_cutoff_date().strftime("%Y-%m-%d")
    query = f"""
        SELECT *
        FROM "OMOP".person p
        WHERE
            p.person_id not in (
                SELECT p.person_id
                FROM
                    "OMOP".visit_occurrence v
                        RIGHT JOIN
                    "OMOP".person p
                        ON p.person_id = v.person_id
                GROUP BY p.person_id, CAST(CAST(p.year_of_birth AS varchar) + '-' + CAST(p.month_of_birth AS varchar) + '-' + CAST(p.day_of_birth AS varchar) AS DATETIME)
                HAVING MIN(visit_start_date) < CAST(CAST(p.year_of_birth AS varchar) + '-' + CAST(p.month_of_birth AS varchar) + '-' + CAST(p.day_of_birth AS varchar) AS DATETIME)
            ) AND
            p.person_id not in (
                SELECT d.person_id
                FROM
                    "OMOP".visit_occurrence v
                        RIGHT JOIN
                    "OMOP".death d
                        ON d.person_id = v.person_id
                GROUP BY d.person_id, d.death_date
                HAVING MAX(visit_start_date) > d.death_date
            ) AND
            p.person_id in (
                SELECT vis.person_id
                FROM "OMOP".visit_occurrence vis
                GROUP BY vis.person_id
                HAVING COUNT( DISTINCT vis.visit_occurrence_id) >= 10
            )


    """
    person = dbf.query(query)
    person.to_csv('./omop/person.csv')


def create_dataset():
    tables = [
        'condition_occurrence',
        'death',
        'visit_occurrence'
    ]
    tables_dates = {
        'condition_occurrence': 'condition_start_date',
        'death': 'death_datetime',
        'visit_occurrence': 'visit_start_date'
    }
    cutoff_date = get_cutoff_date().strftime("%Y-%m-%d")

    for table in tables:
        query = f"""
            SELECT *
            FROM "OMOP".{table} tab
            WHERE
                tab.{tables_dates[table]} < '{cutoff_date}'

        """
        data = dbf.query(query)
        data.to_csv(f"./omop/{table}.csv", index=False)




if __name__ == '__main__':
    select_person()
    create_dataset()
