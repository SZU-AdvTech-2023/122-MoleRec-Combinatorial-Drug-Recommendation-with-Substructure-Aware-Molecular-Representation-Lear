import pickle

with open('history.pkl', 'rb') as f:
    result = pickle.load(f)

print(result)
