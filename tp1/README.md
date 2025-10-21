# TP1 : Fondations du Deep Learning

Ce dossier contient l'implémentation de base d'un réseau de neurones pour la classification des chiffres manuscrits MNIST.

## Installation

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

## Utilisation

```bash
python train_model.py
```

Le script va :
1. Charger les données MNIST
2. Construire un réseau de neurones simple
3. Entraîner le modèle
4. Évaluer ses performances
5. Sauvegarder le modèle entraîné

## Structure du Code

Le réseau de neurones comprend :
- Une couche d'entrée (784 neurones)
- Une couche cachée (512 neurones avec activation ReLU)
- Une couche de dropout (0.2)
- Une couche de sortie (10 neurones avec activation softmax)