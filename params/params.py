from sys import path
path.append('../tools')
from tools import count_params, count_params_reduced

def l_to_s(l):
    return str(l)[1:-1]

with open('./structures.txt', 'r') as f:
    s = []
    for line in f.readlines():
        if line[0] == 'u':
            split_index = line.find('l')
            units_string = line[1:split_index]
            layers_string = line[split_index+1:-1]

            u = [int(x) for x in units_string.split(',')]
            l = [int(x) for x in layers_string.split(',')]
            
            s.append({'Units': l_to_s(u), 'Layers': l_to_s(l), 'Count': count_params_reduced(u, l)})
        else:
            l = [int(x) for x in line.rstrip().split(',')]
            s.append({'Layers': l_to_s(l), 'Count': count_params(l)})
    
with open('./param_counts.txt', 'w') as f:
    for i in range(len(s)):
        g = str(s[i])[1:-1].replace('\'', '') 
        f.write(g + '\n')


