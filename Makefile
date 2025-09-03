remove_punct=True
remove_stopwords=True
content_only=False
metadata_only=False
model_is_tree=False

help:
	@echo "------ 👌 Commandes disponibles :"
	@echo "> make help        - Afficher cette aide"
	@echo "> make requirements - Installer les dépendances"
	@echo "> make clean       - Nettoyer les fichiers temporaires"
	@echo "> make clean_data  - Nettoyer les données préprocessées (force reprocessing)"
	@echo "> make reinstall_package  - Réinstaller package"
	@echo "> make train model_name=        - Entraîner le modèle (avec preprocessing intégré)"
	@echo "> make evaluate model_name=    - Evaluer le modèle"
	@echo "> make run_all model_name=      - run in the order : train -> evaluate"
	@echo " ! Model names: LGBMRegressor, XGBRegressor, GradientBoostingRegressor, Ridge,"
	@echo "   ExtraTreesRegressor, RandomForestRegressor, LinearRegression, ElasticNet"
	@echo ""
	@echo "------ ⚠️  IMPORTANT: Preprocessing is now integrated with training to prevent data leakage"
	@echo "------ ✅ Fin des Commandes."

requirements:
	@echo "------ 🔄 Installation des dépendances..."
	pip install -r requirements.txt
	python -m nltk.downloader all
	@echo "------ ✅ Dépendances installées."

reinstall_package:
	@echo "------ 🔄 Réinstallation du package..."
	@pip install --config-settings editable_mode=compat -e .
	@echo "------ ✅ Package réinstallé."

clean:
	@echo "------ 🧹 Nettoyage des fichiers temporaires..."
	@rm -fr **/__pycache__ **/*.pyc
	@rm -fr **/build **/dist
	@rm -fr medium.egg-info
	@rm -f **/*Zone.Identifier
	@echo "------ ✅ Nettoyage terminé."

clean_data:
	@echo "------ 🧹 Nettoyage des données préprocessées..."
	@rm -f /raw_data/medium/df_*processed*.csv
	@echo "------ ✅ Données préprocessées supprimées."

clean_models:
	@echo "------ 🧹 Nettoyage des modèles et preprocesseurs..."
	@echo "⚠️  ATTENTION: Cette commande va supprimer tous les modèles!"
	@echo "Appuyez sur Ctrl+C pour annuler, ou attendez 5 secondes..."
	@sleep 5
	@rm -f ~/medium/models/*.pickle
	@rm -f ~/medium/preprocessor/*.pickle
	@echo "------ ✅ Modèles et preprocesseurs supprimés."

test:
	@echo "------ 🔄 Start test ..."
	@pytest -v
	@echo "------ ✅ End!"

lint:
	@echo "------ 🔄 Start lint..."
	@echo "To Do 🆘"
	@echo "------ ✅ End lint!"

# DEPRECATED: Preprocessing is now integrated with training to prevent data leakage
# Use 'make train' instead which handles both preprocessing and training
preprocess_deprecated:
	@echo "------ ❌ DEPRECATED: Preprocessing séparé peut causer des fuites de données!"
	@echo "------ ℹ️  Utilisez 'make train' qui gère le preprocessing correctement."
	@echo "------ ℹ️  Le preprocessing est maintenant intégré dans l'entraînement."

train:
	@echo "------ 🔄 Start train (avec preprocessing intégré)..."
	@echo "ℹ️  Configuration:"
	@echo "  - Model: $(model_name)"
	@echo "  - Remove punctuation: $(remove_punct)"
	@echo "  - Remove stopwords: $(remove_stopwords)"
	@echo "  - Content only: $(content_only)"
	@echo "  - Metadata only: $(metadata_only)"
	@echo "  - Model is tree: $(model_is_tree)"
	python -c "from medium.interface.main import train; train(model_name='$(model_name)', remove_punct=$(remove_punct), remove_stopwords=$(remove_stopwords), content_only=$(content_only), metadata_only=$(metadata_only), model_is_tree=$(model_is_tree))"
	@echo "------ ✅ End train."

pred:
	@echo "------ 🔄 Start pred ..."
	python -c "from medium.interface.main import pred; pred(model_name='$(model_name)', text='$(text)')"
	@echo "------ ✅ End pred."

evaluate:
	@echo "------ 🔄 Start evaluate ..."
	python -c "from medium.interface.main import evaluate; evaluate(model_name='$(model_name)', remove_punct=$(remove_punct), remove_stopwords=$(remove_stopwords), content_only=$(content_only), metadata_only=$(metadata_only), model_is_tree=$(model_is_tree))"
	@echo "------ ✅ End evaluate."

run_all:
	@echo "------ 🔄 Run all (train + evaluate)..."
	python -c "from medium.interface.main import run_all; run_all(model_name='$(model_name)', remove_punct=$(remove_punct), remove_stopwords=$(remove_stopwords), content_only=$(content_only), metadata_only=$(metadata_only), model_is_tree=$(model_is_tree))"
	@echo "------ ✅ End run all."

# Validation commands
validate_setup:
	@echo "------ 🔍 Validation de la configuration..."
	@echo "Vérification des preprocesseurs train-only:"
	@ls -la ~/medium/preprocessor/*train_only* 2>/dev/null || echo "Aucun preprocesseur train-only trouvé"
	@echo ""
	@echo "Vérification des données préprocessées:"
	@ls -la ~/medium/data/df_*processed*.csv 2>/dev/null || echo "Aucune donnée préprocessée trouvée"
	@echo "------ ✅ Validation terminée."

# Service API
as_service:
	uvicorn medium.api.fast:app --reload

# Development helpers
watch_metrics:
	@echo "------ 📊 Surveillance des métriques..."
	@watch -n 2 'ls -lht ~/medium/metrics/ | head -10'

compare_models:
	@echo "------ 📈 Comparaison des modèles..."
	@python -c "import pandas as pd; import glob; files = glob.glob('~/medium/metrics/*.csv'); [print(f'{f}: MAE = {pd.read_csv(f)[\"mae\"].mean():.4f}') for f in files[-5:]]"

# Full pipeline with best model
production_ready:
	@echo "------ 🚀 Préparation pour la production..."
	@make clean
	@make train model_name=XGBRegressor
	@make evaluate model_name=XGBRegressor
	@echo "------ ✅ Modèle prêt pour la production!"
