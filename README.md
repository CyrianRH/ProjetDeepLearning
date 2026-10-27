# 🛰️ Deep Learning & NLP Projects (L3)

Projets académiques de Deep Learning menés dans le cadre de la troisième année de Licence (L3).  
Ce dépôt regroupe deux études comparatives complètes :
1. **Classification d'images satellites** : Détection binaire d'éoliennes.
2. **Classification multi-classes de textes** : Prédiction de catégories de métiers (28 classes).

---

## 📑 Sommaire
- [Projet 1 : Détection d'Éoliennes sur Images Satellites](#-projet-1--détection-déoliennes-sur-images-satellites)
- [Projet 2 : Classification de Textes - Profils de Métiers](#-projet-2--classification-de-textes---profils-de-métiers)
- [Contraintes & Environnement Matériel](#-contraintes--environnement-matériel)
- [Installation & Utilisation](#-installation--utilisation)
- [Rapports & Supports](#-documentation)

---

## 🛰️ Projet 1 : Détection d'Éoliennes sur Images Satellites

### Contexte & Données
- **Objectif** : Classification binaire (`Target` / `Other`) indiquant la présence ou non d'une éolienne.
- **Dataset** : Images satellites couleur en résolution $128 \times 128$.
  - Entraînement : 58 218 images (20% réservés pour le test interne).
  - Validation : 5 000 images.
  - Test final : 5 000 images.

### Modèles & Résultats

| Modèle | Paramètres | Précision (Val) | Précision (Test Interne) | Temps / Époque |
| :--- | :---: | :---: | :---: | :---: |
| **Régression Logistique** | 49 000 | 85,84% | 85,87% | ~15 s |
| **CNN Simple** (1 conv 5x5 + Dense) | 7 900 000 | 93,34% | 94,10% | ~25 s |
| **CNN Profond** (32 → 64 → 128 + Dropout) | 560 000 | 96,86% | 97,28% | ~75 s |
| **EfficientNetV2 Small** (Fine-tuning) | 20 000 000 | 96,80% | 97,20% | ~150 s |
| **MobileNetV3** (Fine-tuning) | **3 000 000** | **97,18%** | **97,43%** | **~70 s** |

### Enseignements clés
- **MobileNetV3** s'est révélé être le modèle le plus efficient, alliant légèreté architecturale et précision maximale.
- **Analyse du jeu de test final** : Alors que les performances atteignaient ~97% en validation et test interne, la précision chute à ~90% sur l'ensemble de test inconnu. Cette divergence s'explique par des variations de distribution géographique/capteur ainsi que par des ambiguïtés d'étiquetage sur des infrastructures à peine perceptibles.

---

## 📚 Projet 2 : Classification de Textes - Profils de Métiers

### Contexte & Données
- **Objectif** : Classification multi-classes prédisant le métier (28 catégories) à partir d'une description textuelle anglophone issue de Common Crawl.
- **Défi majeur** : Fort déséquilibre de classes (la classe professorat représente plus d'un tiers des données) et recoupement lexical entre métiers proches.
- **Prétraitement** : Nettoyage regex, lemmatisation, suppression des stopwords, et validation croisée sur 5 folds.

### Modèles & Comparatif

| Modèle | Représentation / Dimension | Paramètres | Précision | Poids de classes | Temps / Époque |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Naïf Bayes** | Bag-of-Words (10k voc.) | 280 000 | 71,13% | Non | ~3 s |
| **Naïf Bayes** | TF-IDF (10k voc.) | 280 000 | 69,76% | Non | ~3 s |
| **MLP (Réseau Dense)** | BOW + SVD (300 dims) | 100 000 | 76,00% | Non | ~10 s |
| **MLP (Réseau Dense)** | **TF-IDF + SVD (300 dims)** | **100 000** | **77,29%** | Non | **~10 s** |
| **LSTM (64 unités)** | Embeddings aléatoires | 1 000 000 | 32,25% → **76,27%** | Non → **Oui** | ~30 s |
| **LSTM (64 unités)** | Word2Vec pré-entraîné | 1 000 000 | 74,45% | Oui | ~30 s |

### Enseignements clés
- **Défaillance du LSTM initial** : Sans pondération, le LSTM converge vers une solution triviale prédisant uniquement la classe majoritaire (32,25% d'accuracy). L'introduction d'une fonction de perte pondérée (`class_weights`) a débloqué l'apprentissage (76,27%).
- **Compromis Performance / Coût** : L'approche hybride **TF-IDF + SVD + Réseau Dense** surpasse les réseaux récurrents tout en étant 3 fois plus rapide à converger.
- **Plafond de verre (~77%)** : La superposition lexicale de certains intitulés (ex. *Data Scientist* vs *Software Engineer*) montre les limites des embeddings statiques et oriente les perspectives vers des Transformers (ex. BERT).

---

## ⚙️ Contraintes & Environnement Matériel

Les entraînements ont été réalisés en local avec accélération matérielle :
- **GPU** : Nvidia GeForce GTX 1070
- **Framework** : TensorFlow 2.10 (version cible assurant la compatibilité CUDA/cuDNN avec l'architecture GPU)
- **Outils** : Keras, Scikit-learn, Pandas, NLTK, Gensim

---

## 🚀 Installation & Utilisation

1. **Cloner le projet** :
   ```bash
   git clone [https://github.com/](https://github.com/)<votre-pseudo>/<nom-du-repo>.git
   cd <nom-du-repo>
