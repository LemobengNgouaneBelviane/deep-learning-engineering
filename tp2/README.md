# Deep Learning Engineering Practice

Ce projet contient les travaux pratiques de Deep Learning Engineering, focalisés sur le développement, 
le suivi et le déploiement de modèles de Deep Learning.

## Structure du Projet

```
.
├── requirements.txt     # Dépendances Python
├── train_model.py      # Script d'entraînement avec MLflow
├── app.py             # API Flask pour le déploiement
└── Dockerfile         # Configuration Docker
```

## Installation

```bash
# Créer un environnement virtuel
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
.\venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt
```

## Utilisation

### Entraînement du Modèle

```bash
python train_model.py
```

### Lancement de l'API

```bash
python app.py
```

### Construction et Lancement du Container Docker

```bash
docker build -t mnist-api .
docker run -p 5000:5000 mnist-api
```

## Test de l'API

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"image": [0, 0, ..., 0]}'  # tableau 1D de 784 valeurs
```

## Auteur
[Votre Nom]

## Licence
MIT