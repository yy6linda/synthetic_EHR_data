# synthetic_EHR_data
This repository records the work done for synthetic ehr project
## generating 1% dataset
Run retrieving_files.py, this generates a 1% omop dataset.
## extract features from the 1 % dataset
Run process_omop, this converts the omop condition table to a pivoted table where each row is a patient, and each columns is a medical code.
```python
python process_omop.py condition_occurrence.csv condition binary 
```
