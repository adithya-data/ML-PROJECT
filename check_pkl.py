import pickle

try:
    with open("productivity_model.pkl", "rb") as f:
        artifact = pickle.load(f)
    
    print("Keys in artifact:", artifact.keys())
    if "features" in artifact:
        print("Features count:", len(artifact["features"]))
        print("Features:", artifact["features"])
    else:
        print("Features key NOT FOUND")
    
    if "scaler" in artifact:
        print("Scaler found")
    else:
        print("Scaler key NOT FOUND")
        
    if "model" in artifact:
        print("Model type:", type(artifact["model"]))

except Exception as e:
    print("Error loading artifact:", e)
