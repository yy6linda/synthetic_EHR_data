import pandas as pd
import numpy as np
import random
'''modify visit'''

def visit_occurrence():
    '''given a patient's visit records, this function returns the prediction_date '''
    '''and whether this patient has a death record (1) or not(0)'''
    '''output is a reduced visit file'''
    person = pd.read_csv('./person.csv')
    visit = pd.read_csv('./visit_occurrence.csv')
    cols = ['person_id','visit_start_date']
    visit = visit[cols]
    visit = visit[visit.person_id.isin(person.person_id)]
    death = pd.read_csv('./death.csv')
    cols = ['person_id','death_date']
    death = death[cols]
    visit_death = pd.merge(death,visit,on = ['person_id'],how = 'inner')
    visit_death['death_date'] = pd.to_datetime(visit_death['death_date'], format = '%Y-%m-%d')
    visit_death['visit_start_date'] = pd.to_datetime(visit_death['visit_start_date'], format = '%Y-%m-%d')
    visit_death['last_visit_death'] = visit_death['death_date'] - visit_death['visit_start_date']
    visit_death['last_visit_death'] = visit_death['last_visit_death'].apply(lambda x: x.days)
    visit_death = visit_death.loc[visit_death['last_visit_death'] <= 180]
    visit_death = visit_death.drop_duplicates(subset = ['person_id'], keep = 'first')
    visit_death = visit_death[['person_id','visit_start_date']]
    visit_death.to_csv('./omop_5x/visit_death_all.csv', index = False)
    visit_live = visit[~visit.person_id.isin(visit_death.person_id)]
    visit_live = visit_live[['person_id','visit_start_date']]
    visit_live = visit_live.sort_values(['person_id','visit_start_date'],ascending = False).groupby('person_id').head(1)
    visit_live = visit_live[['person_id','visit_start_date']]
    #live_id = visit_live[['person_id']].drop_duplicates(keep = 'first')
    '''
    for patients in the negative case, select patients' latest visit record
    '''
    death_count = visit_death.shape[0]
    live_count = death_count * 5
    random.seed(0)
    '''This is used to generate a random 5x live patient set'''
    random_number = random.randint(live_count,visit_live.shape[0])-live_count
    visit_live = visit_live.iloc[random_number:(random_number + live_count),:]
    visit_live.to_csv('./omop_5x/visit_live_5x.csv', index = False)
    return visit_live, visit_death

def person(visit_live, visit_death):
    person = pd.read_csv('./person.csv')
    person_live = person[person.person_id.isin(visit_live.person_id)]
    person_death = person[person.person_id.isin(visit_death.person_id)]
    person_live.to_csv('./omop_5x/person_live_5x.csv', index = False)
    person_death.to_csv('./omop_5x/person_death_all.csv', index = False)

def condition(visit_live, visit_death):
    condition = pd.read_csv('./condition_occurrence.csv')
    condition_live = condition[condition.person_id.isin(visit_live.person_id)]
    condition_death = condition[condition.person_id.isin(visit_death.person_id)]
    condition_live.to_csv('./omop_5x/condition_live_5x.csv', index = False)
    condition_death.to_csv('./omop_5x/condition_death_all.csv', index = False)

if __name__ == '__main__':
    visit_live, visit_death =visit_occurrence()
    person(visit_live, visit_death)
    condition(visit_live, visit_death)
