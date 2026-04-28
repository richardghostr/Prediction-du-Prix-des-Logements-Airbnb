# Airbnb Price Prediction — README

Prédiction du logarithme du prix (`log_price`) d'un logement Airbnb à partir de données hétérogènes (textuelles, numériques, catégorielles, géographiques, temporelles).

## Contenu du dépôt
- `airbnb_prediction.ipynb` — Notebook principal (EDA, FE, modélisation, optimisation, génération de `prediction.csv`).
- `airbnb_train.csv` — Jeu d'entraînement (contient `log_price`).
- `airbnb_test.csv` — Jeu de test pour inférence (sans `log_price`).
- `prediction_example.csv` — Format attendu pour la soumission.

## Objectif
Construire un pipeline reproductible qui prédit `log_price`. Le notebook effectue :
- EDA et visualisations
- Feature engineering (amenities, texte, dates, flags)
- Encodage et imputation
- Entraînement et comparaison de modèles (Linear, Ridge, Lasso, RandomForest, GradientBoosting)
- Optimisation par `RandomizedSearchCV`
- Génération du fichier `prediction.csv` au format requis

## Installation (rapide)
Prérequis : Python 3.8+, pip, Jupyter

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

## Lancer le notebook
1. Placer les fichiers CSV dans le même dossier que le notebook.
2. Lancer Jupyter :

```bash
jupyter notebook airbnb_prediction.ipynb
# ou
jupyter lab
```
3. Kernel → Restart & Run All.

> Le notebook génère `prediction.csv` à la fin (colonnes `Unnamed: 0,logpred`).

## Aperçu rapide du pipeline
1. Chargement des données (`pd.read_csv`).
2. EDA : distributions, valeurs manquantes, corrélations, analyses par catégorie et géographiques.
3. Feature engineering :
   - Parsing `amenities` → variables binaires + `amenities_count`
   - Dates (`first_review`, `last_review`, `host_since`) → jours écoulés + flag `has_reviews`
   - Text features : `description_length`, `name_length`, `premium_keyword_count`
   - Booléens/flags → conversion en 0/1
   - Ratio `beds_per_bedroom`
4. Imputation : mediane pour numériques, `Unknown` pour `neighbourhood/zipcode`.
5. Encodage : `pd.get_dummies` sur `property_type`, `room_type`, `bed_type`, `cancellation_policy`, `city` (concat train+test puis dummies).
6. Standardisation pour modèles linéaires (Ridge/Lasso/LinearRegression).
7. Entraînement et évaluation (RMSE, MAE, R²), CV 5-fold sur le meilleur modèle.
8. Optimisation : RandomizedSearchCV sur GradientBoosting.

## Remarques importantes et recommandations
- La cible est `log_price`. Si votre modèle prédit le prix brut P, appliquez `np.log(P)` avant de remplir `prediction.csv`.
- Conversion des flags : attention aux valeurs non-standard ("t", "f", True/False, "True"/"False"). Utiliser mapping robuste avant .astype(int).
- `pd.get_dummies` sur colonnes à forte cardinalité (ex. `neighbourhood`, `city`) peut augmenter fortement la dimension. Envisager : target encoding, réduction des modalités rares (regrouper en `Other`) ou fréquence minimale.
- Pipeline reproductible : fitter imputer/encoders/scaler sur le train et réutiliser pour test (utiliser `Pipeline` / `ColumnTransformer`).
- Vérifier l'existence et la colonne d'ID du test : le notebook utilise parfois `Unnamed: 0`; préférez une colonne `id` explicite.

## Format de soumission
Le fichier de soumission doit respecter exactement le format de `prediction_example.csv` :

```
Unnamed: 0,logpred
14282777,4.781464
17029381,4.781464
...
```
- `Unnamed: 0` : identifiant du logement (colonne index du test)
- `logpred` : prédiction de `log_price` (float)

## Commandes utiles
- Extraire un échantillon du test (si le fichier est volumineux) :

Python :
```python
import pandas as pd
pd.read_csv('airbnb_test.csv', nrows=500).to_csv('airbnb_test_sample.csv', index=False)
```

PowerShell :
```powershell
Get-Content .\airbnb_test.csv -TotalCount 501 | Set-Content .\airbnb_test_sample.csv
```

## Modèles testés (dans le notebook)
- Linear Regression (baseline)
- Ridge (L2)
- Lasso (L1)
- Random Forest Regressor
- Gradient Boosting Regressor
- Optimisation du GB via `RandomizedSearchCV`

## Résultats observés (exemples)
- Linear / Ridge / Lasso : RMSE ≈ 0.57–0.58
- Random Forest : RMSE ≈ 0.45
- Gradient Boosting : RMSE ≈ 0.43
- GB optimisé : RMSE ≈ 0.42

> Ces valeurs sont indicatives et dépendent des runs et des échantillonnages.

## Pistes d'amélioration
- NLP avancé sur `description` et `name` : TF-IDF, embeddings (sentence-transformers).
- XGBoost / LightGBM : souvent plus performants et plus rapides que le GB scikit-learn.
- Target encoding pour `neighbourhood` et autres catégories à haute cardinalité.
- Features géographiques : clustering spatial, distances aux centres urbains, ou variables agrégées par zone.
- Stacking / blending de modèles pour réduire variance.

## Checklist pour rendu
- [ ] `airbnb_prediction.ipynb` — Notebook exécuté avec outputs visibles
- [ ] `prediction.csv` — généré dans le bon format
- [ ] EDA complète avec visualisations
- [ ] Comparaison d'au moins 4 modèles et métriques RMSE / MAE / R²
- [ ] Importance des features affichée
- [ ] Validation croisée & optimisation des hyperparamètres documentées

## FAQ rapide
Q: Mon modèle prédit le prix brut — que faire ?
A: Appliquer `np.log()` sur les prédictions avant d'écrire `logpred`.

Q: L'exécution prend trop de temps ?
A: Réduire `n_estimators` (100–150) et `n_iter` pour `RandomizedSearchCV` (10 itérations) pour accélérer.

Q: Problèmes de colonnes entre train et test ?
A: Assurez-vous d'appliquer `pd.get_dummies` sur la concaténation `pd.concat([X_train, X_test])` puis de séparer.

---

Bonne utilisation — si vous voulez, je peux :
- Générer une version courte en anglais.
- Lancer des correctifs sur `airbnb_prediction.ipynb` (p.ex. robustifier conversions booléennes et imputation via `ColumnTransformer`).

