import pandas as pd
conv = pd.read_csv("dic_icd_omop.csv",header = 0, names=["condition_source_value", "condition_concept_id"])

print("dictionary created")
i = 0
final = pd.DataFrame()
for chunk in pd.read_csv("condition_occurrence.csv", chunksize=100000):
	i = i + 1
	chunk.drop(columns=['condition_concept_id'],inplace =True)
	chunk = chunk.merge(conv, on ="condition_source_value", how = 'inner')
	print("chunk{} processed".format(i),flush = True)
	if i == 1:
		final = pd.concat([chunk, final])
	else:
		final = pd.concat([final,chunk])
final.to_csv("condition_occurrence_converted.csv", index = False)
