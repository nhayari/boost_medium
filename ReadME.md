# 📊 Projet de Prédiction des Recommandations d'Articles Medium

## 📋 Description du Projet :
	Ce projet vise à prédire le nombre de recommandations que recevront les articles publiés sur Medium.
	Précisemment prédire la variable cible log1p_recommends (logarithme du nombre de recommandations + 1) à partir des caractéristiques des articles.

## 🎯 Objectifs :
  Analyser les données des articles Medium
  Développer un modèle de prédiction des recommandations
  Atteindre la meilleure performance possible avec la métrique MAE (Mean Absolute Error)


## ⚙️ Installation et Configuration
###  Prérequis
  Python 3.10.6+

### Installation
- Cloner le repository:
  git clone <repository-url>
  cd BOOST_MEDIEM

- Les commandes disponibles :
  make help

- Installer les dépendances :
  make requirments

- Copier le fichier d'exemple
  cp .env.example .env

- Éditer le fichier .env avec vos chemins :
  nano .env    (ou utiliser votre éditeur favori)

- Chemins des fichiers de données
   - TRAIN_JSON=path/train.json
   - TEST_JSON=path/test.json
   - TARGET_CSV=path/train_log1p_recommends.csv

- Créer les dossiers suivant
  ```bash
  mkdir -p ~/medium/data/
  mkdir -p ~/medium/params/
  mkdir -p ~/medium/models/
  mkdir -p ~/medium/preprocessor/
  mkdir -p ~/medium/metrics/
  ```


### 🚀 Utilisation

Commandes principales:
  - Installation et setup
      make setup           : Configuration initiale
			make requirments     : Installation des dépendances

		- Développement
			make reinstall_package : Réinstaller le package après modifications
			make clean             : Nettoyer les fichiers temporaires

		- Charger les données
			df = load_json_from_files(TRAIN_PATH, num_lines=1000)


### 📊 Données
- Fichiers requis
    train.json : Données d'entraînement (62,313 articles)
    test.json : Données de test (34,645 articles)
    train_log1p_recommends.csv : Variable cible pour l'entraînement

- Structure des données:
    Chaque article contient :
      _id : Identifiant unique
      url : URL de l'article
      published : Date de publication
      title : Titre de l'article
      author : Informations sur l'auteur
      content : Contenu HTML de l'article
      meta_tags : Métadonnées supplémentaires


### 🤝 Contribution
  - Forker le projet
  - Créer une branche feature (git checkout -b mybranch)
	- Add modifications (git add .)
	- Committer les changes (git commit -m 'my modif')
  - Pusher sur la branche (git push origin mybranch)
  - Ouvrir une Pull Request

### 📝 License
 Ce projet est sous licence MIT.

### 🙏 Remerciements
  Team Medium
  LeWagon

### 📞 Support
	Pour toute question ou problème, veuillez ouvrir une issue sur le repository GitHub.
