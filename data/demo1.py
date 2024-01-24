import pickle

with open('substructure_smiles.pkl', 'rb') as f:
    data = pickle.load(f)
print(data)
print(len(data))
