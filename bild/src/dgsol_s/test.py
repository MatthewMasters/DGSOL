#dgsol test

import numpy as np
import torch
import os
import subprocess

from tqdm import tqdm
a=[[1,2,3],[4,5,6],[7,8,9],[10,11,12],[13,14,15]]
elements=np.array(['C','C','C','C','C'])
a=np.array(a)*0.1
a=torch.from_numpy(a)
dm=torch.cdist(a,a)
print(dm)
indices=[]
lower_traingle_indices=torch.tril_indices(5,5,-1).T
dgsol='/data/exp/DGSOL/bild/src/dgsol_s/dgsol_s'

def solve_dgp(outpath, n_solutions=10):
    cmd = f'{dgsol} -s{n_solutions} {outpath}/dgsol.input {outpath}/dgsol.output {outpath}/dgsol.summary'
    process = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE)
    output, error = process.communicate()
    if error is not None:
        print(outpath, error)
    
def parse_dgsol_errors(outpath):
    """
    There are 4 types of errors in the dgsol output:

    f_err      The value of the merit function
    derr_min      The smallest error in the distances
    derr_avg      The average error in the distances
    derr_max      The largest error in the distances
    """
    with open(f'{outpath}/dgsol.summary', 'r') as input:
        lines = input.readlines()

    errors = []
    # skip the header lines
    for line in lines[5:]:
        errors.append(line.split()[2:])   # the first two entries are n_atoms and n_distances
    return np.array(errors).astype('float32')
    
def output_to_xyz(outpath, solution_id, n_solutions, elements):
    n_atoms = elements.shape[0]
    with open(f'{outpath}/dgsol.output') as outfile:
        lines = outfile.readlines()

    coords = []
    for line in lines:
        if not line.startswith('\n') and len(line) > 30:
            coords.append([float(n) for n in line.split()])
    coords = np.array(coords).reshape(n_solutions, n_atoms, 3)
    t=torch.from_numpy(coords)
    print(torch.cdist(t[0],t[0]))
    
    coords.append(coords[solution_id])

    with open(f'{outpath}/dgsol.xyz', 'w') as outfile:
        outfile.write(f'{n_atoms}\n')
        outfile.write('\n')
        for xyz, element in zip(coords[solution_id], elements):
            outfile.write(f'{num_to_element[int(element)]}\t{xyz[0]}\t{xyz[1]}\t{xyz[2]}\n')
    
def to_scientific_notation(number):
    a, b = '{:.17E}'.format(number).split('E')
    num = '{:.12f}E{:+03d}'.format(float(a) / 10, int(b) + 1)
    return num[1:]

def solve_distance_geometry(outpath, n_solutions=10):
    construction_errors = []

    for i, ids in tqdm(enumerate(np.arange(len(self.distances))), total=len(self.distances)):
        out = f'{outpath}/{ids:04}'
        os.makedirs(out, exist_ok=True)
        write_dgsol_input(distances=self.distances[i], outpath=out)
        solve_dgp(out, n_solutions=n_solutions)
        errors = parse_dgsol_errors(out)
        lowest_errors_idx = np.argsort(errors[:, 2])
        construction_errors.append(errors[lowest_errors_idx[0]])
        self.output_to_xyz(out,
                       solution_id=lowest_errors_idx[0],
                       n_solutions=n_solutions,
                       elements=self.elements[i])
    c_errors = np.array(construction_errors)

def write_dgsol_input(distances, outpath):
    n, m = np.triu_indices(distances.shape[1], k=1)
    with open(f'{outpath}/dgsol.input', 'w') as outfile:
        for i, j in zip(n, m):
            outfile.write(
            f'{i + 1:9.0f}{j + 1:10.0f}   {to_scientific_notation(distances[i, j])}   '
            f'{to_scientific_notation(distances[i, j])}\n')


    
out='/data/exp/DGSOL/bild/src/dgsol_s/ex'
construction_errors=[]
write_dgsol_input(dm,out)
#solve_dgp(out, n_solutions=3)
errors = parse_dgsol_errors(out)
lowest_errors_idx = np.argsort(errors[:, 2])
construction_errors.append(errors[lowest_errors_idx[0]])
output_to_xyz(out,lowest_errors_idx[0],3,elements)
print('amr')
