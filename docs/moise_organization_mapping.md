# Organisation CORA et vocabulaire MOISE+

Référence : Hübner, Sichman et Boissier, *A Model for the Structural, Functional, and Deontic Specification of Organizations in Multiagent Systems*, SBIA 2002. Le PDF fourni est la référence de notation.

| Concept | Sens retenu |
| --- | --- |
| Objectif | État ou résultat attendu, avec des conditions d’atteinte à définir. |
| Mission | Ensemble cohérent d’objectifs auquel un agent s’engage. |
| Rôle | Position dans une structure organisationnelle ; distincte de l’agent qui l’adopte. |
| Groupe | Structure regroupant des rôles et leurs relations ; instances peuplées par les agents. |
| Schéma social | Décomposition d’un objectif collectif, plans et missions associées. |
| Permission / obligation | Relation déontique entre rôle et mission, conditionnée temporellement. |
| Spécification organisationnelle (OS) | Dimensions structurelle, fonctionnelle et déontique. |
| Entité organisationnelle (OE) | Organisation instanciée : agents jouant des rôles et engagés dans des missions. |

Une action du simulateur n’est pas un objectif. Une mission n’est pas un synonyme de rôle. Une récompense du simulateur ne prouve pas à elle seule l’atteinte d’un objectif MOISE+.

## Notation et vues temporelles

Les groupes sont des boîtes à onglet, les rôles des boîtes arrondies et les compositions groupe–rôle portent un losange ouvert du côté du groupe, avec leurs cardinalités. Les objectifs sont nommés dans les arbres fonctionnels, avec missions en exposant et barres de plan selon la notation du package. Les plans peuvent être séquentiels (virgule), alternatifs (barre verticale) ou parallèles (∥). Les missions regroupent des objectifs. Les permissions et obligations appartiennent à la vue déontique ; une obligation implique une permission.

Les liens d’accointance, communication et autorité sont des relations entre rôles. Les lignes colorées agent–mission ajoutées dans notre vue temporelle représentent des engagements et constituent une surcouche explicite, distincte de ces liens MOISE+.

## Limites du prototype

`src/cora/organization.py` encode actuellement les rôles des agents, les affectations de mission, des permissions booléennes et une version. Les interventions natives permutent principalement les engagements de mission. Elles ne changent pas automatiquement les rôles.

Les groupes, schémas sociaux complets, obligations temporelles et prédicats d’atteinte des objectifs ne sont pas intégralement exécutés par ce prototype. Les dessins présentent une projection compatible avec ce vocabulaire, pas une certification de conformité MOISE+ complète. Les interfaces du calcul et de la campagne en cours restent inchangées.

`paper/organization/generate.py` produit le schéma des dimensions et l’évolution des engagements à partir d’une trace existante. `organization_history.json` consigne la source, son empreinte, les affectations et les instants de changement. Aucun nouvel entraînement n’est nécessaire. Les sources et limites pédagogiques sont également consignées dans `AI-disclosure.md`.

Régénération : `.venv/bin/python paper/organization/generate.py`.
