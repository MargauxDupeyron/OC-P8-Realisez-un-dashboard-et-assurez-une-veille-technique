## Architecture du système

1. Le dashboard interroge l’API de scoring via HTTP.
2. L’API renvoie :
   - la probabilité de défaut
   - la décision
   - le seuil utilisé
   - les valeurs SHAP locales
3. Le dashboard interprète et visualise ces informations
   de manière intelligible pour un public non technique.
