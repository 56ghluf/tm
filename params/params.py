from sys import path
path.append('../tools')
from tools import count_params

with open('./structures.txt', 'r') as f:
    s = []
    for line in f.readlines():
        s.append([int(x) for x in line.rstrip().split(',')])
    
    counts = []
    
    for structure in s:
        counts.append(count_params(structure))

with open('./param_counts.txt', 'w') as f:
    for i in range(len(counts)):
        f.write(f'{str(s[i])[1:-1]} : {counts[i]}\n')


