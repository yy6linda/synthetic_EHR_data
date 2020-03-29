import pandas as pd

procedure= pd.read_csv('procedure_occurrence.csv')
procedure_concept = procedure['procedure_concept_id'].drop_duplicates()
procedure_concept.to_csv('./filter/procedure_occurrence_concepts.csv', index = False)

drug = pd.read_csv('drug_exposure.csv')
drug_concept = drug['drug_concept_id'].drop_duplicates()
drug_concept.to_csv('./filter/drug_exposure_concepts.csv', index = False)

measurement = pd.read_csv('measurement.csv')
measurement_concept = measurement['measurement_concept_id'].drop_duplicates()
measurement_concept.to_csv('./filter/measurement_concepts.csv', index = False)

observation= pd.read_csv('observation.csv')
observation_concept = observation['observation_concept_id'].drop_duplicates()
observation_concept.to_csv('./filter/observation_concepts.csv', index = False)





