# CORA : attentes et conception des rendus

12 octobre 2026. Analyse conditionnelle des hypothèses et des configurations locales `bench_research.yaml`, `applications.yaml` et `applications_adaptive.yaml`. Les configurations et versions réellement exécutées sur le serveur doivent être comparées à celles-ci avant interprétation. Ce document ne modifie ni les expériences ni les seuils de sélection.

## Ce que le setup permet de tester

Le mécanisme recherché est : des interventions potentiellement complémentaires → un modèle d'effets pertinent → des expériences qui discriminent les meilleures décisions → une adoption tenant compte des coûts et de l'incertitude. Une trajectoire agréable ne montre pas, à elle seule, que cette chaîne fonctionne.

Les problèmes à effets connus portent sur six modifications, avec cinq budgets de 4 à 64 et dix graines. Ils sont le principal support pour tester le bénéfice d'une restriction correcte des interactions. Les environnements natifs utilisent toutes les interactions : ils n'établissent donc pas une réduction de dimension par localité. Les budgets natifs vont de 4 à 32 ; l'évaluation dure 25 pas avec actualisation 0,99, un coût de 0,1 par permutation et 64 répétitions de référence. Les six politiques incluent une politique aléatoire et cinq apprenants. Les 40 400 pas demandés ne permettent pas de présumer la convergence. Les trois rounds adaptatifs réentraînent entre décisions ; ils ne constituent pas un protocole de perturbation suivi de récupération.

Les permutations sont ordonnées et atomiques. Des vecteurs de modifications distincts peuvent mener à la même affectation, avec des coûts différents : comparer aussi les transactions et les organisations finales évite de confondre redondance et exploration utile.

## Attentes quantitatives

| Question | Tendance si le mécanisme est utile | Rendu conseillé | Cas qui limite ou contredit l'attente |
|---|---|---|---|
| Regret simple | Diminution globale avec le budget ; avantage à budget limité sur les deux contrôles appariés | IQM et intervalle bootstrap par graine ; différences appariées CORA−contrôle | Plateaux, intersections, différence compatible avec zéro ; CORA peut rester au statu quo |
| Erreur d'effet hors échantillon | Réduction avec davantage de requêtes ; avantage de composition lorsque les interactions pertinentes sont correctement représentées | Erreur contre budget, séparée par régime ; nuage effet estimé/référence avec diagonale | Graphe incomplet : biais persistant ; dynamique dense : bénéfice structural absent |
| Efficacité des interventions | Moins de requêtes pour franchir un même seuil de regret, lorsque celui-ci est accessible | Fraction des graines atteignant le seuil à chaque budget | Ne pas attribuer le budget maximal aux échecs comme s'ils avaient franchi le seuil ; première traversée peut être suivie d'une rechute |
| Harm rate | Moins de décisions à valeur nette inférieure au no-op qu'avec adoption gloutonne | Risque parmi adoptions, fréquence d'adoption et valeur nette côte à côte | Aucune adoption : taux conditionnel indéfini ; abstention universelle ne montre pas une adaptation utile |
| Abstention | Souvent élevée à petit budget, puis susceptible de diminuer lorsque les différences deviennent informatives | Proportion no-op contre budget, avec raisons de refus | Coût trop fort, interventions équivalentes ou absence de gain : abstention durable rationnelle |
| Couverture diagnostique | À mesurer, avec erreurs particulièrement possibles sous misspécification/changement de politique | Couverture contre budget, par régime | Le rayon configuré n'est pas une garantie à 95 % ; un intervalle bootstrap d'IQM est un objet différent |
| Contraintes | Zéro violation pour les méthodes utilisant le validateur, sous invariants représentés | Table de violations avec nombre d'essais | Respect partagé avec les contrôles : ne discrimine pas CORA ; obligations plus riches non représentées |
| Coût de calcul | Acquisition active souvent plus coûteuse par requête ; réduction éventuelle du budget total utile | Regret contre requêtes et contre temps ; décomposition entraînement/requêtes/référence/rendu | Si le gain de requêtes disparaît en temps total, annoncer le compromis |
| Retraining entre rounds | Effets et classements peuvent évoluer après mise à jour des politiques | Round × méthode, séparément pour chaque environnement/apprenant | Trois rounds ne démontrent ni convergence ni temps de récupération après panne |

### Forme des courbes et statistiques

Une tendance décroissante ne doit pas être forcée à être monotone. Une nouvelle observation bruitée, une estimation déplacée ou un changement d'intervention sélectionnée peuvent augmenter ponctuellement le regret. Avec dix graines, les intervalles peuvent être larges, irréguliers et se chevaucher. Le bruit ne doit pas être ajouté aux résultats mesurés pour leur donner une apparence particulière.

Distinguer les incertitudes : dispersion des rollouts, variabilité entre graines, erreur Monte-Carlo des références et intervalle du résumé statistique. Les checkpoints de budget d'une même exécution ne sont pas indépendants. Conserver les unités natives ; n'agréger entre domaines qu'avec une normalisation définie explicitement et stable. Une amélioration de 0,1 unité n'a pas le même sens dans tous les environnements.

Un chiffre analytique fixe peut être donné : à horizon 25 et gamma 0,99, la somme des poids est environ 22,22. Elle convertit une borne connue de récompense par pas en borne de retour, mais ne prédit aucun score appris. Les effets des permutations et les écarts entre méthodes ne sont pas déductibles numériquement des hypothèses seules. En particulier, le coût de 0,1 signifie qu'une permutation ne paie pas si son gain de retour attendu est inférieur à 0,1 ; plusieurs modifications doivent couvrir leur coût cumulé.

Les références à 64 répétitions sont des estimations. L'optimum estimé sur beaucoup de candidats subit un effet de sélection ; les écarts très proches doivent être interprétés avec l'incertitude des références. Un no-op affiché par CORA peut être la bonne décision si aucun gain net n'est identifiable.

## Attentes par contexte

### Problèmes à effets connus

**Structure clairsemée correcte.** Attendre une meilleure réutilisation de chaque expérience entre plusieurs compositions. La séparation entre CORA, factorisation uniforme et acquisition sans structure est le test important ; une victoire sur l'exploration aléatoire seule est insuffisante. Le modèle additif peut plafonner si les interactions sont fortes. Un réseau global peut bien fonctionner à grand budget et reste un contrôle utile.

**Structure dense.** Les courbes peuvent se rapprocher. Le coût d'acquisition peut dépasser l'économie de requêtes. Une famille restrictive erronée peut perdre contre une estimation sans structure.

**Dépendances manquantes.** Voir des compositions mal prédites, des résidus systématiques et éventuellement une couverture faible malgré une apparente confiance. Une heatmap estimation−référence par composition rend mieux cette défaillance qu'une simple moyenne.

**Changement de politique.** Un modèle accurate pour la référence peut devenir mal calibré au déploiement. Montrer deux colonnes de prédictions/valeurs avec versions de politique et la dérive des classements. La concentration statistique sous politique fixe ne résout pas cette dérive.

### World Comm

Quatre poursuivants sont contrôlés, dont un leader ; deux foragers suivent un comportement fixé. L'intervention permute des consignes de secteur. Si les poursuivants utilisent ces consignes, une bonne affectation pourrait réduire les poursuites redondantes et améliorer la couverture des directions de fuite ; une coordination via les communications du leader serait un signe comportemental pertinent. Ces attentes sont conditionnelles : une simple permutation de consignes peut ne pas induire une telle spécialisation.

À montrer : positions et traces des poursuivants, secteurs assignés en couleur, messages réellement émis par le leader et retour moyen poursuivants. Afficher séparément le retour des foragers ; un total des deux équipes peut masquer l'objectif. Ne pas interpréter chaque rapprochement comme une capture ou chaque couleur comme une mission effectivement suivie.

### SMAX 3m

Trois alliés affrontent des ennemis heuristiques. L'intervention permute des consignes de cible, sans forcer le choix de l'attaque. Des politiques suffisamment compétentes pourraient concentrer les dégâts, achever une cible avant d'en changer et limiter les pertes. Une stratégie différente peut néanmoins être préférable selon positions, portée et cooldowns.

À montrer : santé et statut des unités, cible assignée et cible de l'action réellement exécutée, disponibilité des actions et cooldowns. Utiliser des instantanés aux décisions ; ne pas inventer des tirs entre frames. Le focus fire est un comportement possible de l'apprenant, pas la preuve que CORA l'a causé. Comparer la même politique avant/après la permutation.

La symétrie des marines et les politiques partagées peuvent rendre les permutations peu utiles. Les masques d'actions disponibles sont traités différemment entre PPO et les méthodes Q ; ne pas attribuer cet effet à CORA.

### OvercookedV2

Deux chefs, une permutation élémentaire, consignes préparation/cuisson et service. Si elles sont apprises, attendre une répartition plus complémentaire des trajets, moins de blocage mutuel et moins d'attente lors de la cuisson. Les ingrédients, plats, livraison et occupation du pot offrent des événements visibles.

Cependant, les politiques ne sont pas contraintes par ces consignes. Des agents symétriques déjà bien coordonnés peuvent ne rien gagner à les échanger. Le coût peut alors rendre no-op optimal. Avec seulement deux candidats, les méthodes devraient souvent converger rapidement vers le même choix : ce benchmark ne peut pas établir un gain de composition.

Le horizon 25 peut terminer avant une livraison. Un score faible ou nul n'implique pas nécessairement absence de préparation. Montrer les états du pot/inventaires et les événements effectivement présents. Les replays existants de 110 pas et leur planificateur à état complet ne sont pas équivalents au protocole d'évaluation des politiques apprises.

### JaxNav

Trois robots, objectifs réellement permutés. C'est l'intervention native la plus directement visible : une affectation préférable pourrait réduire les détours, les croisements ou la contention. Les cartes, collisions et points de départ peuvent rendre l'affectation la plus courte individuellement mauvaise collectivement.

À montrer : liens robot→objectif avant/après, trajectoires, distances restantes, obstacles, LiDAR et événements de collision/arrivée. Séparer ces diagnostics du retour net : les termes de récompense peuvent produire un classement différent d'une appréciation visuelle. Une politique entraînée sur ses objectifs initiaux peut mal réagir aux nouveaux objectifs.

Ne pas utiliser le replay pédagogique à deux robots BlankCross2 comme observation de la campagne à trois robots. Les overlays d'affectation doivent venir de l'organisation effectivement appliquée, et les événements des états sauvegardés.

## Idéation : rendre visible le mécanisme de CORA

### 1. Tableau de décision synchronisé — rendu principal recommandé

Gauche : environnement natif. Centre : organisation courante et candidats plausibles, avec missions/ressources. Droite : valeur nette estimée, rayon diagnostique et score du no-op. Bande inférieure : chronologie des requêtes et adoption/refus. Tous les panneaux utilisent la même version de politique et le même temps.

Le moment révélateur est la requête qui départage deux transactions : indiquer la paire concernée, la réduction calculée de l'incertitude de leur différence, puis l'intervention choisie. Un gain de récompense n'est affiché qu'après observation. Lorsqu'un refus est justifié, montrer clairement les intervalles qui se chevauchent et le maintien du statu quo.

### 2. Comparaison de décision, politique figée

À partir du même contexte initial, juxtaposer no-op, décision CORA et décision d'un contrôle organisationnel, avec la même continuation policy. Afficher les coûts et retours cumulés et faire apparaître le instant de divergence. Les futures variables aléatoires doivent être couplées selon un protocole déclaré ; après divergence, les trajectoires ne sont pas forcément identiques hors intervention.

Ce rendu isole mieux le décideur que CORA+MAPPO contre aléatoire ou contre un autre apprenant. La comparaison d'apprenants et celle des réorganisations sont deux figures distinctes.

### 3. Écran « pourquoi cette expérience ? »

Dessiner les deux candidats plausibles, leurs différences de features et les expériences disponibles. Surligner le score d'acquisition minimal calculé. Après la requête, montrer le changement de classement et de marge d'adoption. Si la meilleure expérience n'apporte pas d'information utile, la figure révèle aussi l'échec de l'acquisition.

### 4. Composition : no-op / modification A / modification B / A+B

Montrer quatre branches à contexte et politique fixes, puis la valeur d'interaction F(A+B)−F(A)−F(B)+F(no-op). Une interaction positive significative rend la complémentarité visible ; une interaction négative montre la concurrence. Les quatre branches doivent être admissibles et les valeurs estimées suffisamment précises. Overcooked ne dispose pas de deux modifications élémentaires : réserver ce rendu aux contextes qui le permettent.

### 5. Graphe causal et carte des erreurs

Organisation symbolique à gauche, facteurs de retour et influences temporelles à droite. Une carte des erreurs par composition accompagne le graphe. Seules les dépendances effectivement spécifiées ou identifiées sont affichées comme telles ; une inférence supposée n'est pas dessinée comme vérité découverte. Pour les native full families, montrer leur densité plutôt que fabriquer une structure clairsemée.

### 6. Fiche de réussite, fiche de refus, fiche d'échec

Présenter au moins un gain net robuste, un refus rationnel et une reconfiguration défavorable, avec même format : contexte, expériences, décision, coût, effet, incertitude, invariant valide. Ces fiches répondent à des questions scientifiques plus fortes qu'un montage de seules scènes favorables.

## Diagnostics nécessaires pour les overlays

| Overlay | Source à utiliser | État actuel |
|---|---|---|
| Affectations et missions | Organisation avant/après commit | Sémantique disponible ; vérifier sa présence dans chaque trace |
| Positions, santé, inventaires, objectifs | Arbre natif sauvegardé | Disponible selon environnement |
| Requêtes et budgets | Logs du décideur | À relier explicitement à la timeline du replay |
| Scores et rayons | Prédictions/diagnostic_radius du décideur | Disponibles pour plusieurs méthodes ; pas un certificat |
| Cible/commande effectivement choisie | Actions sauvegardées et décodage natif | À décoder pour l'environnement concerné |
| Interactions vraies | Oracle à effets connus uniquement | Pas une vérité accessible dans les native full families |
| Congestion, détours, délais de cuisson | Diagnostics dérivés avec définition explicite | Propositions ; ne pas annoncer comme métriques déjà calculées |
| Récupération après panne | Protocole de perturbation dédié | Non établi par les configs trois-rounds actuelles |

## Critère final d'interprétation

Le signe favorable attendu est une meilleure décision nette avec moins d'expériences, à politique, contexte et budget comparables. Les mouvements plus propres ne suffisent pas. Pour que le rendu montre le cœur de la contribution, il doit relier l'intervention validée, la raison de son exploration, l'estimation qui justifie l'adoption et son effet observé. Si les données ne produisent pas cette chaîne, le résultat est inconclusif ou négatif dans ce contexte ; il ne faut pas le remplacer par un comportement imposé pour décider une hypothèse scientifique.
