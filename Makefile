help:
	@echo "------ 👌Commandes disponibles :"
	@echo "> make help        - Afficher cette aide"
	@echo "> make requirments - Installer les dépendances"
	@echo "> make clean       - Nettoyer les fichiers temporaires"
	@echo "> make reinstall_package  - Réinstaller package"
# 	@echo "> make test        - Exécuter les tests"
# 	@echo "> make lint        - Vérifier la qualité du code"
# 	@echo "> make data        - Télécharger les données"
	@echo "> make preprocess       - Traiter les données"
	@echo "> make train model_name=        - Entraîner le modèle"
	@echo "> make evaluate  model_name=    - Evaluer le modèle"
	@echo "> make run_all model_name= remove_punct= remove_stopwords= content_only= metadata_only= model_is_tree=     - run in the order : preprocess -> train -> evaluate"
	@echo "> make pred model_name=yourmodel text=yourtext               - text must be in '  '"
	@echo " ! actually model_name : LGBMRegressor or XGBRegressor or GradientBoostingRegressor or Ridge or ExtraTreesRegressor or RandomForestRegressor or LinearRegression or ElasticNet."
	@echo "------ ✅ Fin des Commandes."


requirments:
	@echo "------ 🔄 Réinstallation du package..."
	pip install -r requirements.txt
	python -m nltk.downloader all
	@echo "------ ✅ Package réinstallé."

reinstall_package:
	@echo "------ 🔄 Réinstallation du package..."
# 	@pip uninstall -y medium || :
# 	@pip install -e .
# delete deprecated warning setup
	@pip install --config-settings editable_mode=compat -e .
	@echo "------ ✅ Package réinstallé."

clean:
	@echo "------ 🧹 Nettoyage..."
# 	@rm -f */version.txt
# 	@rm -f .coverage
	@rm -fr **/__pycache__ **/*.pyc
	@rm -fr **/build **/dist
	@rm -fr medium.egg-info
# 	@rm -f **/.DS_Store
	@rm -f **/*Zone.Identifier
	@echo "------ ✅ Nettoyage terminé."



test:
	@echo "------ 🔄 Start test ..."
# 	@echo "To Do 🆘"
	@pytest -v
	@echo "------ ✅ End!"

lint:
	@echo "------ 🔄 Start lint..."
	@echo "To Do 🆘"
	@echo "------ ✅ End lint.!"

data:
	@echo "------ 🔄 Start data load..."
	@echo "To Do 🆘 if necessary"
	@echo "------ ✅ End data load."

preprocess:
	@echo "------ 🔄 Start preprocess ..."
	python -c "from medium.interface.main import preprocess; preprocess(remove_punct=$(remove_punct), remove_stopwords=$(remove_stopwords), content_only=$(content_only), metadata_only=$(metadata_only), model_is_tree=$(model_is_tree))"
	@echo "------ ✅ End preprocess."

train:
	@echo "------ 🔄 Start train ..."
# 	python -c 'from medium.interface.main import train; train()'
	python -c "from medium.interface.main import train; train(model_name='$(model_name)', remove_punct=$(remove_punct), remove_stopwords=$(remove_stopwords), content_only=$(content_only), metadata_only=$(metadata_only), model_is_tree=$(model_is_tree))"
	@echo "------ ✅ End train."

pred:
	@echo "------ 🔄 start pred ..."
	@echo "To Do 🆘"
	python -c "from medium.interface.main import pred; pred(model_name='$(model_name)', text='$(text)')"
	@echo "------ ✅ End pred."

evaluate:
	python -c "from medium.interface.main import evaluate; evaluate(model_name='$(model_name)', remove_punct=$(remove_punct), remove_stopwords=$(remove_stopwords), content_only=$(content_only), metadata_only=$(metadata_only), model_is_tree=$(model_is_tree))"

run_all:
	@echo "------ 🔄 run all ..."
# 	python -c 'from medium.interface.main import run_all; run_all()'
		python -c "from medium.interface.main import run_all; run_all(model_name='$(model_name)', remove_punct=$(remove_punct), remove_stopwords=$(remove_stopwords), content_only=$(content_only), metadata_only=$(metadata_only), model_is_tree=$(model_is_tree))"
	@echo "------ ✅ End run all."

# workflow:
# 	PREFECT__LOGGING__LEVEL=${PREFECT_LOG_LEVEL} python -m medium.interface.workflow

as_service:
uvicorn medium.api.fast:app --reload
