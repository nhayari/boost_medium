#!/bin/bash

# Script pour créer l'arborescence Medium
echo "Création de l'arborescence Medium..."

# Création des dossiers
# cd ~/content/
mkdir -p ~/medium/data/
mkdir -p ~/medium/params/
mkdir -p ~/medium/models/
mkdir -p ~/medium/preprocessor/
mkdir -p ~/medium/metrics/
mkdir -p ~/medium/prediction/

# Vérification et confirmation
echo "Arborescence créée avec succès !"
echo "Structure :"
tree ~/medium/ 2>/dev/null || ls -la ~/medium/
