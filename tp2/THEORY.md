# Réponses aux Questions Théoriques - TP2

## Question 1 : Couches Dense et Dropout

### Couche Dense
La couche `Dense` est une couche entièrement connectée où chaque neurone est connecté à tous les neurones de la couche précédente. Dans notre modèle :
- La première couche Dense (512 neurones) transforme les entrées de dimension 784 (28x28 pixels) en représentations de plus haut niveau
- L'activation ReLU permet d'introduire de la non-linéarité, nécessaire pour apprendre des motifs complexes
- La couche de sortie (10 neurones) correspond aux 10 classes possibles (chiffres 0-9)

### Dropout
La couche `Dropout` est une technique de régularisation qui :
- Désactive aléatoirement 20% des neurones pendant l'entraînement
- Force le réseau à apprendre des caractéristiques plus robustes
- Réduit la dépendance à des neurones spécifiques
- Agit comme un ensemble de réseaux, améliorant la généralisation

### Softmax
La fonction d'activation `softmax` est utilisée dans la couche de sortie car :
- Elle convertit les sorties en probabilités (somme = 1)
- Chaque valeur représente la probabilité d'appartenance à une classe
- Idéale pour la classification multi-classes (10 chiffres dans notre cas)

## Question 2 : Optimiseur Adam

Adam (Adaptive Moment Estimation) améliore la SGD classique en :
1. Adaptant les taux d'apprentissage pour chaque paramètre
2. Combinant les avantages de :
   - RMSprop : adaptation aux gradients récents
   - Momentum : accélération dans la bonne direction
3. Utilisant des estimations du premier moment (moyenne) et du second moment (variance non centrée) des gradients
4. Auto-ajustant les pas d'apprentissage, nécessitant moins de réglage manuel

## Question 3 : Vectorisation et Calculs par Lots

### Vectorisation
- Les images sont aplaties de (28,28) à (784,) pour traitement vectoriel
- Les opérations sont effectuées sur des matrices plutôt que élément par élément
- Utilisation de numpy et des opérations tensorielle de TensorFlow

### Calculs par Lots (Batch Processing)
- `batch_size=128` : traite 128 images simultanément
- Avantages :
  - Meilleure utilisation du parallélisme matériel (CPU/GPU)
  - Estimation plus stable des gradients qu'avec un seul exemple
  - Compromis entre mise à jour fréquente des poids et stabilité