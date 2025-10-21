# Travaux Pratiques Deep Learning Engineering

Ce dépôt contient l'implémentation des deux TPs de Deep Learning Engineering, focalisés sur le développement,
le suivi et le déploiement de modèles de Deep Learning.

## Structure du Projet

Le projet est organisé en deux parties distinctes :

### TP1 : Fondations du Deep Learning
- Implémentation d'un réseau de neurones de base pour MNIST
- Concepts fondamentaux (couches denses, dropout)
- Évaluation des performances

### TP2 : Ingénierie et Déploiement
- Améliorations du modèle (L2, BatchNorm)
- Suivi des expériences avec MLflow
- API REST avec Flask
- Conteneurisation avec Docker
- Pipeline CI/CD

## Guide de Démarrage

### Pour le TP1 :
```bash
cd tp1
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python train_model.py
```

### Pour le TP2 :
```bash
cd tp2
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Entraînement avec MLflow
python train_model.py

# Lancement de l'API
python app.py

# Construction de l'image Docker
docker build -t mnist-api .
docker run -p 5000:5000 mnist-api
```

## Documentation

- TP1 : Voir `tp1/README.md` pour les détails de l'implémentation de base
- TP2 : 
  - `tp2/README.md` pour l'utilisation avancée
  - `tp2/THEORY.md` pour les explications théoriques
  - `tp2/CICD.md` pour la configuration du pipeline CI/CD
