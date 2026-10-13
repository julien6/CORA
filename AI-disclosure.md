
# Anticipation des résultats et idéation visuelle — 12 octobre 2026

Une assistance générative a servi à analyser les configurations locales, relier les hypothèses aux métriques et proposer des rendus dans `docs/expected_behavior_and_visual_design.md`. Les comportements favorables envisagés sont des attentes conditionnelles, pas des observations issues de l'entraînement sur le serveur. Aucun chiffre de performance natif n'a été inventé pour cette analyse. Les rendus proposés ne sont pas tous implémentés ; certains nécessitent des diagnostics ou un protocole supplémentaire.

Les tableaux, courbes et valeurs déjà présents dans `paper/pedagogical/` relèvent d'un exercice fictif : ils sont construits à partir de paramètres choisis et de perturbations pseudo-aléatoires, sans calibration empirique. Les screenshots et GIFs proviennent de simulations natives avec contrôleurs manuels/heuristiques et ne constituent pas des résultats appris de CORA. Les bandes des courbes représentent les quantiles de graines construites, et non des intervalles expérimentaux. La suppression de bandeaux dans les pixels ne change pas cette provenance. `paper/pedagogical/MEDIA_PROVENANCE.md` et les manifestes conservent les détails.

Le serveur étant déjà en cours d'exécution, ce travail ne constitue pas une préinscription antérieure aux expériences. Il ne doit justifier ni arrêt sélectif, ni retrait d'environnements défavorables, ni modification du protocole pour confirmer des attentes. Toute conclusion empirique devra être fondée sur les sorties effectivement mesurées et leur incertitude. Les auteurs restent responsables de vérifier les affirmations et les sources.

## Présentation visuelle progressive — 12 octobre 2026

`presentations/cora_visual/` est une présentation pédagogique assistée par IA, conçue pour un public non expert. Sa dernière diapositive est « AI-disclosure ». Un problème fini à trois modifications utilise des valeurs choisies, du bruit contrôlé et les vrais sélecteurs logiciels ; leur exécution dans ce cas ne prouve pas une supériorité expérimentale sur les benchmarks natifs. Les données, courbes et animation de décision sont calculées, pas inventées après exécution. Aucun entraînement MARL n'est effectué.

Les comparaisons natives réutilisent les huit traces de contrôleurs manuels/heuristiques contre actions aléatoires. Leurs animations ne sont pas relabelisées comme CORA, et les scores défavorables sont conservés. Les sources et accès privilégiés sont indiqués dans la présentation et les fichiers de provenance. Les résultats du serveur restent à importer lorsqu'ils seront disponibles ; aucune conclusion positive générale n'est préemptée.

## Vues organisationnelles MOISE+

Le schéma des trois dimensions est un exemple conceptuel du goûter, construit pour expliquer la notation MOISE+ de 2002. Il ne décrit pas une expérience mesurée. L’animation `paper/organization/organization_evolution.gif` utilise les affectations réellement enregistrées dans une trace de démonstration logistique à reconfiguration prédéfinie ; cette reconfiguration n’a pas été choisie par CORA. La représentation du groupe est une projection graphique ; les liens agent–mission constituent une surcouche distincte des relations entre rôles. Aucun objectif atteint ni obligation temporelle satisfaite n’est inventé. Le prototype ne constitue pas une implémentation complète de MOISE+.
