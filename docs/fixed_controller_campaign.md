# Campagne centrale à contrôleurs fixes

Cette campagne isole la sélection de changements d'engagement de mission. Aucun entraînement de politique ni réseau de valeur n'est effectué. Elle complète les campagnes natives existantes ; elle ne modifie pas une campagne serveur déjà en cours.

```bash
# Installer notamment le moteur de diagrammes :
.venv/bin/python -m pip install '.[diagrams]'
# Plan uniquement :
./run_fixed_baselines.sh --dry-run
# Vérification courte, dossier neuf :
./run_fixed_baselines.sh --smoke --output outputs/fixed-check
# Campagne complète, uniquement lorsque souhaité :
./run_fixed_baselines.sh --output outputs/fixed-missions
```

## Protocole

Huit agents exécutent huit missions. Quatre interventions permutent chacune les engagements de deux agents. Tous les agents jouent le même rôle, autorisé sur toutes les missions. Les permutations conservent la couverture bijective. Les contrôleurs suivent des règles fixes de demande de ressource et d'arbitrage tournant. Les agents ont des probabilités de service hétérogènes ; les missions demandent différentes quantités de service avant une livraison.

Dans le scénario sparse, les deux groupes de quatre agents ont des ressources indépendantes. Les modifications 0 et 1 ne peuvent influencer que le premier module ; 2 et 3 le second. Le retour collectif est la somme des livraisons. À horizon et tirages exogènes identiques, cette factorisation élimine les interactions entre modules. Les contrôleurs et missions ne sont pas conçus pour garantir une supériorité de CORA.

Le scénario dense partage les ressources entre modules et utilise la famille complète. Le scénario misspecified conserve les dynamiques sparse mais retire les interactions du modèle public. Les interventions affectent explicitement les demandes et capacités de service, sans simple hint facultatif.

Chaque essai repart du même contexte, avec un tirage exogène indépendant par étape d'expérimentation, partagé entre méthodes lorsque les indices d'essai coïncident. Les références utilisent des flux séparés et restent hors de l'interface `PublicProblem`. Les valeurs sont estimées par Monte Carlo ; les échantillons et erreurs standard sont conservés. Les coûts valent 0.15 par permutation. Les rayons sont diagnostiques : aucune garantie certifiée n'est revendiquée.

La campagne normale utilise dix graines, budgets 4/8/16/32, horizon 30 et 64 répétitions de référence. Le smoke utilise une graine, budgets 2/4, horizon 6 et deux répétitions ; il vérifie l'exécution et ne fournit pas de preuve comparative.

## Comparaisons

CORA, structure factorisée avec essais uniformes, modèle tabulaire actif, modèle additif, CORA sans marge d'incertitude et maintien. `linear_g_optimal` minimise à chaque étape la variance prédictive maximale après une observation, avec les mêmes caractéristiques, contraintes, estimateur et règle d'adoption que CORA. C'est un comparateur myope générique, pas une reproduction de RAGE ni un algorithme asymptotiquement optimal revendiqué. Les baselines publiées les plus proches restent à intégrer fidèlement.

## Artifacts et limites

Le dossier contient CSV, statistiques avec bootstrap par graine (aucun intervalle calculé pour une graine), courbes, manifest, observations, décisions, références et replays appariés. Les replays de CORA et G-optimal pour la première graine sparse ont un GIF de télémétrie et une capture. Ces diagrammes sont des vues d'un simulateur de files de ressources, pas des images d'un entrepôt spatial. Les sources `.moise` et vues SVG avant/après reflètent les engagements enregistrés. Les journaux incluent le rang, les estimations, marges et décision ; le scénario dense ou un budget insuffisant peut donner un résultat nul.

Le moteur refuse un dossier non vide : aucune reprise automatique n'est encore fournie pour cette nouvelle campagne. Les campagnes natives et leur reprise restent distinctes. Une validation native non triviale et une reproduction publiée sur caractéristiques identiques sont nécessaires avant de revendiquer une validation complète de CORA.
