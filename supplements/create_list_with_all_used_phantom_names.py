import glob

PATH = r"C:\final_data_fluence/"
DATA_SETS = ["training", "test"]

phantoms = []

for ds in DATA_SETS:
    files = glob.glob(PATH + ds + "/*.npz")
    for file in files:
        p_name = file.split("/")[-1].split("\\")[-1].split("_")[0]
        if p_name not in phantoms:
            phantoms.append(p_name)

print(len(phantoms))
print(phantoms)


