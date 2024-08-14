from math import ceil
from numpy import empty, array, vectorize, exp, split, empty_like

### Input and output generation
# Function to convert number to binary string
def _bin_str_from_num(num, length):
    num_str = str(bin(num)).split('b')[-1]
    if len(num_str) < length:
        num_str = (length - len(num_str)) * '0' + num_str
    return num_str

# Generate the inputs and outputs
def gen_inputs_outputs(R):
    S = ceil(R/2) + 1
    
    # Create the inputs and outupts
    raw_inputs = array(range(2**R), dtype=int)
    
    # Create the empty inputs string
    inputs_str = []

    # Populate inputs_str
    for num in raw_inputs:
        num_str = _bin_str_from_num(num, R)
        inputs_str.append((num_str[0:R//2], num_str[R//2:]))

    # Generate outputs
    outputs = empty([len(inputs_str), S], dtype=int)
    for i, nums in enumerate(inputs_str):
        outputs[i] = array([int(char) for char in _bin_str_from_num(int(nums[0], 2) + int(nums[1], 2), S)])
   
    # Convert inputs_str
    inputs = empty([len(inputs_str), R], dtype=int)

    for i, inp in enumerate(inputs_str):
        a = [int(x) for x in inp[0]]
        a.extend([int(x) for x in inp[1]])    
        inputs[i] = array(a) 

    return inputs, outputs, S

# Creates the dict for all the input units
# takes the size of stacked input layer
# size must be even if not pairs impossible
def _gen_input_unit_dict(l):
    if l % 2:
        raise Exception('Input length must be even.')

    a = dict()

    for i in range(int(l/2)):
        a[f'unit_{i}'] = empty((2**l, 2), dtype=int)
        
    return a

# Convert stacked input to paired input
def gen_pair_inputs(inputs):
    # Get input length
    l = len(inputs[0])
    # Get dictionary to fill
    a = _gen_input_unit_dict(l) 

    # Seperate the input into halves
    first_half = inputs[:,:int(l/2)] 
    second_half = inputs[:, int(l/2):]

    # Empty array to fill with mixed inputs 
    mixed_inputs = empty_like(inputs)
   
    # Itereate and mix
    for i in range(len(inputs)):
        for j in range(0, l, 2):
            mixed_inputs[i][j] = first_half[i][int(j/2)]
            mixed_inputs[i][j+1] = second_half[i][int(j/2)]
    
    # Split to fill dict
    split_mixed = split(mixed_inputs, l/2, axis=1)
    
    # Fill dict with the split
    for i in range(len(split_mixed)):
        a[f'unit_{i}'] = split_mixed[i]
    
    # Finish up
    return a

### Transfer function
# Convert from binary input to symetrical input 
def _hardsym(x):
    return 2*x - 1

hardsym = vectorize(_hardsym)

# Hardlimit 
def _hardlim(x):
    if x >= 0:
        return 1
    return 0

hardlim = vectorize(_hardlim)

# Hardlimit symetrical
def _hardlims(x):
    if x >= 0:
        return 1
    return -1

hardlims = vectorize(_hardlims)

# Log-sigmoid
def _logsig(x):
    return 1/(1+exp(-x))

logsig = vectorize(_logsig)

# Linear
def _lin(x):
    return x

lin = vectorize(_lin)

### Derivatives of the transfer functions
# Hard limit
def _deriv_hardlim(x):
    return 0

deriv_hardlim = vectorize(_deriv_hardlim)

# Linear
def _deriv_lin(x):
    return 1

deriv_lin = vectorize(_deriv_lin)

# Log-sigmoid
def _deriv_logsig(x):
    return (1-_logsig(x)) * _logsig(x)

deriv_logsig = vectorize(_deriv_logsig)

### Error calculation
def e2(E):
    err = 0
    for i in range(len(E)):
        for j in range(len(E[0])):
            err += E[i][j]**2
    return err

### Count the number of parameters
def count_params(l, inp=16, out=9):
    count = 0
    
    ps = inp

    for s in l:
        count += s * ps + s
        ps = s

    count += out * ps + out 

    return count

if __name__ == '__main__':
    print(count_params(2, 2))
