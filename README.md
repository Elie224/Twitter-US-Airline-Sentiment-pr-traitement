# Twitter US Airline Sentiment - Pretraitement

Ce projet contient un notebook de pretraitement pour le dataset `Tweets (2).csv` (Tweets US Airline Sentiment).

## Contenu du projet

- `Tweets (2).csv` : dataset source.
- `sentiment_pipeline.ipynb` : notebook principal avec les etapes de preparation et de visualisation.
- `sentiment_pipeline.py` : version script Python de la logique principale.

## Objectif

Realiser la preparation des donnees pour une classification de `airline_sentiment`.

Le notebook couvre les points suivants :

1. Chargement du dataset.
2. Analyse de base avec pandas (dimensions, types, valeurs manquantes, distribution cible).
3. Selection des variables utiles pour la classification.
4. Encodage de `airline_sentiment` en entiers (`negative=0`, `neutral=1`, `positive=2`).
5. Separation en jeux train/validation et test (split stratifie).
6. Construction d un pipeline avec :
   - suppression des variables inutiles,
   - fonction maison de pretraitement des tweets.

## Lancer le notebook

1. Ouvrir le dossier dans VS Code.
2. Selectionner l environnement Python (`.venv`).
3. Ouvrir `sentiment_pipeline.ipynb`.
4. Executer les cellules dans l ordre.

## Dependances

- pandas
- scikit-learn
- matplotlib

Installation (si necessaire) :

```bash
pip install pandas scikit-learn matplotlib
```
