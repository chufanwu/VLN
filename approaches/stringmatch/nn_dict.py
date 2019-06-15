# construct NN dictionary
def read_json(filename):
    with open(filename, "r") as jfile:
        content = json.load(jfile)
        return content
        
def all_nn_vocab():
    with open(final_)