import sys
import _pickle as pickle
import numpy as np
from datetime import datetime
print('Hi')
if __name__ == '__main__':
    conditionFile = sys.argv[1]
    outFile = sys.argv[2]
    binary_count = sys.argv[3]

    if binary_count != 'binary' and binary_count != 'count':
        print('You must choose either binary or count.')
        sys.exit()


    person_condition = {}
    '''condition_type is a list of unique concept_id'''
    condition_type = []
    '''condition_index is {434354: 2}, [concept_id, order of concept_id]'''
    condition_index = {}
    person = []

    infd = open(conditionFile, 'r')
    infd.readline()
    i = 0
    for line in infd:
        i = i+1
        print('line number: {}'.format(i))
        tokens = line.strip().split(',')
        person_id = int(tokens[1])
        condition = int(tokens[2])

        if condition not in condition_type:
            condition_index[condition] = len(condition_type)
            condition_type.append(condition)
        if person_id not in person:
            person.append(person_id)
        if person_id in person_condition:
            person_condition[person_id].append(condition)
        else:
            person_condition[person_id] = [condition]
    infd.close()
    #print(person_condition)

    #print('Constructing the matrix')
    num_person = len(person)
    num_condition = len(condition_type)
    matrix = np.zeros((num_person, num_condition)).astype('float32')
    for i, person_id in enumerate(person):
        for code in person_condition[person_id]:
            #print(person_id)
            #print(code)
            code_index = condition_index[code]
            if binary_count == 'binary':

                #print(i)
                #print(code_index)
                #print("''''")
                matrix[i][code_index] = 1
            else:
                matrix[i][code_index] += 1
    #print(matrix)
    pickle.dump(matrix, open(outFile+'.npy', 'wb'), -1)
    with open ('./condtion_type.csv', 'w') as f:
        for key, value in condition_index.items():
            text = str(key)+','+str(value)+'\n'
            f.writelines(text)
