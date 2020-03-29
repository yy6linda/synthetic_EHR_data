import pandas as pd
import argparse
from datetime import datetime
import dateutil.relativedelta
import os
import database_functions as dbf

def get_cutoff_date():
    query = """
        SELECT MAX(death_datetime) as date
        FROM omop.death
    """
    output = dbf.query(query)

    date = output["date"][0].date()

    cutoff = date - dateutil.relativedelta.relativedelta(months=6)

    return cutoff

def filter_dataset():
    tables = [
        'person',
        'condition_occurrence',
        'death',
        'visit_occurrence',
        'drug_exposure',
        'measurement',
        'observation',
        'observation_period',
        'procedure_occurrence'
    ]

    for table in tables:
        query = f"""
            SELECT *
            FROM omop.{table} tab
            WHERE
            NOT EXISTS
                (
                    SELECT p.person_id
                    FROM
                        omop.visit_occurrence v
                            RIGHT JOIN
                        omop.person p
                            ON p.person_id = v.person_id
                    WHERE tab.person_id=p.person_id
                    GROUP BY p.person_id, CAST(CAST(p.year_of_birth AS varchar) + '-' + CAST(p.month_of_birth AS varchar) + '-' + CAST(p.day_of_birth AS varchar) AS DATETIME)
                    HAVING MIN(visit_start_date) < CAST(CAST(p.year_of_birth AS varchar) + '-' + CAST(p.month_of_birth AS varchar) + '-' + CAST(p.day_of_birth AS varchar) AS DATETIME)
                ) AND
            NOT EXISTS
                (
                    SELECT d.person_id
                    FROM
                        omop.visit_occurrence v
                            RIGHT JOIN
                        omop.death d
                            ON d.person_id = v.person_id
                    WHERE tab.person_id=d.person_id
                    GROUP BY d.person_id, d.death_date
                    HAVING MAX(visit_start_date) > d.death_date
                ) AND
            EXISTS
                (
                    SELECT v.person_id
                    FROM omop.visit_occurrence v
                    WHERE tab.person_id=v.person_id
                    GROUP BY v.person_id
                    HAVING COUNT( DISTINCT v.visit_occurrence_id) >= 10
                );
        """
        output = dbf.chunked_query(query, 1000000)
        for chunk in output:  # each chunk is a dataframe
            chunk.to_csv("./data/{}.csv".format(table), index = False,sep=",", mode="a")



if __name__ == '__main__':
     filter_dataset()
