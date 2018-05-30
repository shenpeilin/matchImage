import pickle

def load_model(file_name, ncomps=6):
    return pickle.load(open(file_name))

