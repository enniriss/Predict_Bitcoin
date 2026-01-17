from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

# Chemins absolus vers les modèles
MODEL_PATH = "c:/Users/Utilisateur/Documents/Cour/Data Science/GROUPE-EXAM-FINAL/Predict_Bitcoin/src/models/ecommerce_model.pkl"
KMEANS_MODEL_PATH = "c:/Users/Utilisateur/Documents/Cour/Data Science/GROUPE-EXAM-FINAL/Predict_Bitcoin/src/models/kmeans_model.pkl"

# Chargement des modèles au démarrage
with open(MODEL_PATH, "rb") as f:
    clf = pickle.load(f)
with open(KMEANS_MODEL_PATH, "rb") as f:
    kmeans = pickle.load(f)

app = FastAPI()

class ClientData(BaseModel):
    Education: str
    Marital_Status: str
    Income: float
    Kidhome: int
    Teenhome: int
    Recency: int
    MntWines: float
    MntFruits: float
    MntMeatProducts: float
    MntFishProducts: float
    MntSweetProducts: float
    MntGoldProds: float
    NumDealsPurchases: int
    NumPurchases: int
    NumWebVisitsMonth: int
    AcceptedCmp3: int
    AcceptedCmp4: int
    AcceptedCmp5: int
    AcceptedCmp1: int
    AcceptedCmp2: int
    Complain: int

@app.post("/predict")
def predict(data: ClientData):
    df = pd.DataFrame([data.dict()])
    pred = clf.predict(df)[0]
    proba = clf.predict_proba(df)[0][1]
    return {"prediction": int(pred), "probability": float(proba)}

# Après avoir analysé les clusters, suppose :
# cluster 0 = "riche", cluster 1 = "pauvre"
CLUSTER_LABELS = {0: "riche", 1: "pauvre"}

@app.post("/cluster")
def cluster(data: ClientData):
    df = pd.DataFrame([data.dict()])
    X_transformed = clf.named_steps["preprocess"].transform(df)
    cluster_idx = kmeans.predict(X_transformed)[0]
    cluster_name = CLUSTER_LABELS.get(cluster_idx, str(cluster_idx))
    return {"cluster": int(cluster_idx), "label": cluster_name}