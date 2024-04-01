import json

class Configuration:

    def __init__(self):

        self.instrument = 'CORELLI'

def load_config(filename):

    return Configuration()

with open('data.json', 'w') as f:
    json.dump(x, f, indent=4)
    
with open('data.json') as f:
    z = json.load(f)
    