help:
	@echo "👌 Commandes disponibles...........:"
	@echo "  make help        - Afficher cette aide"
	@echo "  make requirments - Installer les dépendances"
	@echo "  make clean       - Nettoyer les fichiers temporaires"
	@echo "  make reinstall_package  - Réinstaller package"
# 	@echo "  make test        - Exécuter les tests"
# 	@echo "  make lint        - Vérifier la qualité du code"
# 	@echo "  make data        - Télécharger les données"
	@echo "  make process     - Traiter les données"
	@echo "  make train       - Entraîner le modèle"
	@echo "  make evaluate    - Evaluer le modèle"
	@echo "  make run_all     - run in the order : preprocess -> train -> pred -> evaluate"
	@echo "✅ Fin des Commandes !"

requirments:
	@echo "🔄 Réinstallation du package..."
	pip install -r requirements.txt
	@echo "✅ Package réinstallé!"

reinstall_package:
	@echo "🔄 Réinstallation du package..."
	@pip uninstall -y medium || :
	@pip install -e .
	@echo "✅ Package réinstallé!"

clean:
	@echo "🧹 Nettoyage..."
# 	@rm -f */version.txt
# 	@rm -f .coverage
	@rm -fr **/__pycache__ **/*.pyc
	@rm -fr **/build **/dist
	@rm -fr medium.egg-info
# 	@rm -f **/.DS_Store
	@rm -f **/*Zone.Identifier
	@echo "✅ Nettoyage terminé!"



test:
	@echo "🔄 Start test ..."
	@echo "To Do 🆘"
	@echo "✅ End!"

lint:
	@echo "🔄 Start lint..."
	@echo "To Do 🆘"
	@echo "✅ End lint!"

data:
	@echo "🔄 Start data load..."
	@echo "To Do 🆘 if necessary"
	@echo "✅ End data load!"

process:
	@echo "🔄 Start process ..."
	@echo "To Do 🆘"
	python -c 'from medium.interface.main import preprocess; preprocess()'
	@echo "✅ End preocess!"

train:
	@echo "🔄 Start train ..."
	@echo "To Do 🆘"
	python -c 'from medium.interface.main import train; train()'
	@echo "✅ End train !"

pred:
	@echo "🔄 start pred ..."
	@echo "To Do 🆘"
	python -c 'from medium.interface.main import pred; pred()'
	@echo "✅ End pred !"

evaluate:
	python -c 'from medium.interface.main import evaluate; evaluate()'

run_all:
	python -c 'from medium.interface.main import run_all; run_all()'

# workflow:
# 	PREFECT__LOGGING__LEVEL=${PREFECT_LOG_LEVEL} python -m medium.interface.workflow

# as_service:
# 	uvicorn medium.api.fast:app --reload
