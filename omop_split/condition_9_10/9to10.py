import pandas as pd
conv = pd.read_csv("icd9_10.csv")
conv_dict = dict(zip(conv.ICD9, conv.ICD10))
print("dictionary created")
i = 0
final = pd.DataFrame()
for chunk in pd.read_csv("condition_occurrence.csv", chunksize=100000):
	i = i + 1
	chunk = chunk.replace({"condition_source_value":conv_dict})
	print("chunk{} processed".format(i),flush = True)
	if i == 1:
		final = pd.concat([chunk, final])
	else:
		final = pd.concat([final,chunk])
final.to_csv("condition_occurrence_converted.csv", index = False)
