import sys
import _pickle as pickle
import numpy as np
from datetime import datetime
import csv
'''this takes in two condition_occurrence.csv files respectively for alive
and deceased patients and outputs two matrices for the two cohort.
The output matrix doesn't have headers nor column index'''

if __name__ == '__main__':
    conditionliveFile = sys.argv[1]
    conditiondeathFile = sys.argv[2]
    outFile_live = sys.argv[3]
    outFile_death = sys.argv[4]
    binary_count = sys.argv[5]

    if binary_count != 'binary' and binary_count != 'count':
        print('You must choose either binary or count.')
        sys.exit()


    live_person_condition = {}
    death_person_condition = {}
    '''condition_type is a list of unique concept_id'''
    condition_type = []
    '''condition_index is {434354: 2}, [concept_id, order of concept_id]'''
    condition_index = {}
    live_person = []
    death_person = []

    '''for alive cohorts'''
    print('read condition_alive')
    infd = open(conditionliveFile, 'r')
    infd.readline()
    i = 0
    for line in infd:
        i = i+1
        print('line number: {}'.format(i))
        tokens = line.strip().split(',')
        person_id = int(float(tokens[1]))
        condition = int(float(tokens[2]))

        if condition not in condition_type:
            condition_index[condition] = len(condition_type)
            condition_type.append(condition)
        if person_id not in live_person:
            live_person.append(person_id)
        if person_id in live_person_condition:
            live_person_condition[person_id].append(condition)
        else:
            live_person_condition[person_id] = [condition]
    infd.close()
    #print(person_condition)

    w1 = csv.writer(open("live_person_id_list.csv", "w"))
    w1.writerows(map(lambda x: [x], live_person))

    '''for deceased cohorts'''
    infd = open(conditiondeathFile, 'r')
    infd.readline()
    i = 0
    for line in infd:
        i = i+1
        print('line number: {}'.format(i))
        tokens = line.strip().split(',')
        person_id = int(float(tokens[1]))
        condition = int(float(tokens[2]))
        if condition not in condition_type:
            condition_index[condition] = len(condition_type)
            condition_type.append(condition)
        if person_id not in death_person:
            death_person.append(person_id)
        if person_id in death_person_condition:
            death_person_condition[person_id].append(condition)
        else:
            death_person_condition[person_id] = [condition]
    infd.close()
    w1 = csv.writer(open("death_person_id_list.csv", "w"))
    w1.writerows(map(lambda x: [x], death_person))


    w2 = csv.writer(open("condition_id_list.csv", "w"))
    for key, val in condition_index.items():
        w2.writerow([key, val])

    print('Constructing the matrix for deceased patients ')
    num_person = len(death_person)
    num_condition = len(condition_type)
    matrix = np.zeros((num_person, num_condition)).astype('float32')
    for i, person_id in enumerate(death_person):
        for code in death_person_condition[person_id]:
            code_index = condition_index[code]
            if binary_count == 'binary':
                matrix[i][code_index] = 1
            else:
                matrix[i][code_index] += 1
    #print(matrix)
    np.save(outFile_death,matrix)

    print('Constructing the matrix for alive patients ')
    num_person = len(live_person)
    num_condition = len(condition_type)
    matrix = np.zeros((num_person, num_condition)).astype('float32')
    for i, person_id in enumerate(live_person):
        for code in live_person_condition[person_id]:
            code_index = condition_index[code]
            if binary_count == 'binary':
                matrix[i][code_index] = 1
            else:
                matrix[i][code_index] += 1
    #print(matrix)
    np.save(outFile_live,matrix)

    #pickle.dump(matrix, open(outFile+'.matrix', 'wb'), -1)
