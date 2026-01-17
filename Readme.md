# Utilisat


## Architecture des fichiers
``` 
Predict_Bitcoin/
│
├── Api/
│   └── api.py   (optionnel mais recommandé)
|
├── data/
|   ├── marketing_campaign.csv      <----- Fichier csv brut
|   └── processed/
|
├── src/
|   ├── models/
|   |   ├── ecommerce_model.pkl
|   |   └── kmeans_model.pkl
|   |
|   ├── analyse.ipynb    <-----Notebook avec l'analyse globale
|   |
|   └── Pipeline.py      <-----Fichier avec les fonctions principals servant à l'API
```

## Lancer le projet

Créationn d'un environnement python et activation
``` bash
python -m venv .env
source .env/bin/activate
```

Installation des dépendances
``` bash
pip install -r requirements.txt
```

Faites en sorte lancer le script à la racine du projet, par exemple `.../Predict_Bitcoin/`.

Lancer les Pipelines
``` bash
python src/Pipeline.py
```

Lancer l'api
``` bash
uvicorn Api.api:app --reload
```