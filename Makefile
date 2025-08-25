help:
	@echo "👌 Commandes disponibles...........:"
	@echo "  make help        - Afficher cette aide"
	@echo "  make requirments - Installer les dépendances"
	@echo "  make clean       - Nettoyer les fichiers temporaires"
	@echo "  make reinstall_package  - Réinstaller package"
	@echo "  make test        - Exécuter les tests"
	@echo "  make lint        - Vérifier la qualité du code"
	@echo "  make data        - Télécharger les données"
	@echo "  make process     - Traiter les données"
	@echo "  make train       - Entraîner le modèle"
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
	@echo "🔄 Début ..."
	@echo "To Do 🆘"
	@echo "✅ End!"

lint:
	@echo "🔄 Début ..."
	@echo "To Do 🆘"
	@echo "✅ End!"

data:
	@echo "🔄 Début data load..."
	@echo "To Do 🆘 if necessary"
	@echo "✅ End data load!"

process:
	@echo "🔄 Début process ..."
	@echo "To Do 🆘"
	@echo "✅ End preocess!"

train:
	@echo "🔄 Début train ..."
	@echo "To Do 🆘"
	@echo "✅ End train !"
