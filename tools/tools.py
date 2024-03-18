from math import ceil
from numpy import empty, array, vectorize

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

### Error calculation
def e2(E):
    err = 0
    for i in range(len(E)):
        for j in range(len(E[0])):
            err += E[i][j]**2
    return err

# Hardlimit symetrical
def _hardlims(x):
    if x >= 0:
        return 1
    return -1

hardlims = vectorize(_hardlims)

if __name__ == '__main__':
    print(gen_inputs_outputs(4))
