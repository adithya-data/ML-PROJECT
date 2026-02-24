import pickle
import pandas as pd

with open('productivity_model.pkl', 'rb') as f:
    artifact = pickle.load(f)

print("Features from artifact:")
print(artifact.get('features', 'No features found'))
if 'model' in artifact:
    print("\nModel type:", type(artifact['model']))
