import os
import glob
import json
import numpy as np
path = r"I:\research\seblab\data\group_folders\Janek\PALPAITINE_ZURICH\raw\DIS/"


def read_thicknesses(phantom_name):
    counter = 0
    thicknesses = dict()
    for m_file in glob.glob(path + phantom_name + "/*"):
        if os.path.isdir(m_file):
            thicknesses[counter] = list(np.loadtxt(m_file + "/thickness.txt"))
            counter += 1

    return thicknesses


data_files = glob.glob("data/*.npz")
results = dict()

for data_file in data_files:
    name = data_file.split("/")[-1].split("\\")[-1][:-4]
    print(name)
    phantom_a = name + ".a/"
    phantom_b = name + ".b/"
    results[phantom_a] = read_thicknesses(phantom_a)
    results[phantom_b] = read_thicknesses(phantom_b)

with open(f"data/thickness.json", "w+") as json_file:
    json.dump(results, json_file)
