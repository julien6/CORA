# MOISE Diagram dans CORA

CORA utilise [MOISE Diagram](https://github.com/julien6/MOISE_Diagram), un compilateur de spécifications MOISE+ indépendant. La version 0.2.0 est fixée au commit `0403d820dea1773559e79bb7771d00e8a8016678` dans l'extra `diagrams` du `pyproject.toml` pour reproduire les rendus.

## Installation

Depuis la racine de CORA :

```bash
.venv/bin/python -m pip install '.[diagrams]'
```

Git est nécessaire pour installer la dépendance distante ; Graphviz doit fournir `dot` et `neato` pour les rendus. Le package Python n'a pas de dépendance à CORA ou au MARL. La version et sa provenance sont consignées dans `docs/moise_diagram_dependency.json`.

## Compilation et vues

```bash
.venv/bin/python -m moise_diagram examples/moise/gouter.moise -o organization.svg
.venv/bin/python -m moise_diagram examples/moise/gouter.moise -o structure.pdf --view ss --key
.venv/bin/python -m moise_diagram examples/moise/gouter.moise -o goals.svg --view fs --key
.venv/bin/python -m moise_diagram examples/moise/gouter.moise -o norms.svg --view ds --ds-style matrix --ds-inherited
.venv/bin/python -m moise_diagram examples/moise/gouter.moise -o integrated.svg --view ss --view fs --ds-links --key
```

Le package fournit les vues structurelle (`ss`), fonctionnelle (`fs`), déontique (`ds`) et entité organisationnelle (`oe`). La vue déontique peut être une liste de prédicats, un graphe ou une matrice, avec normes héritées. Les liens déontiques peuvent traverser les vues SS et FS. Le moteur de rendu route les cardinalités et les liens conformément aux conventions documentées dans le package. Utiliser `compile_file` plutôt qu'un simple appel `dot` pour bénéficier de cette étape.

## API Python

```python
from moise_diagram import parse, compile_file

compile_file('examples/moise/gouter.moise', 'organization.pdf',
             views=['ss', 'fs'], key=True, ds_links=True)
model = parse(open('examples/moise/gouter.moise', encoding='utf-8').read())
print(model.ancestors('serveur'))
```

L'entrée historique `cora.moise_diagram` réexporte l'API du package ; le code nouveau utilise directement `moise_diagram`.

## Syntaxe utilisée dans CORA

```text
@moise "Goûter"
role preparateur
role serveur
group gouter
contains gouter preparateur 1..1
contains gouter serveur 1..2
link gouter preparateur serveur com intra
scheme servir pret
 goal pret "Goûter prêt"
 goal biscuits "Biscuits disposés"
 goal verres "Verres disposés"
plan pret = biscuits || verres
mission m_biscuits
mission m_verres
goals m_biscuits biscuits
goals m_verres verres
per preparateur m_biscuits Any
obl serveur m_verres Any
@end
```

Les séquences utilisent `,`, les choix `|`, les plans parallèles `||`. Les plans acceptent également les degrés de succès et des expressions imbriquées. Chaque objectif appartient à un schéma social ; les missions ne peuvent pas mélanger des objectifs de schémas différents. Les liens et compatibilités sont rattachés à un groupe. La [référence du langage](https://github.com/julien6/MOISE_Diagram/blob/0403d820dea1773559e79bb7771d00e8a8016678/LANGUAGE.md) et l'[API](https://github.com/julien6/MOISE_Diagram/blob/0403d820dea1773559e79bb7771d00e8a8016678/docs/api.md) sont maintenues dans le dépôt du package.

## Figures et évolution temporelle

```bash
.venv/bin/python paper/organization/generate.py
```

Cette commande produit les vues SS/FS/DS/OE, la matrice déontique, le graphe des normes et la vue SS–FS intégrée dans `paper/organization/`. Les figures utilisées par le papier et la présentation Beamer sont mises à jour. Les fichiers `.moise` des instantanés temporels et les images correspondantes sont conservés. Un GIF résume les engagements d'une trace existante, sans simuler de nouvelles transitions ni entraîner d'agents.

Les groupes et schémas ajoutés aux instantanés de la trace sont une projection explicative du modèle CORA réduit. Aucun objectif atteint n'est inféré. Le package vérifie notamment les références, les arbres de plans, les compatibilités et les permissions d'engagement. Il n'exécute pas les obligations, n'évalue pas les conditions temporelles ni l'atteinte des objectifs, et ne vérifie pas les effectifs des instances.

Tests d'intégration : `.venv/bin/python -m pytest tests/test_moise_diagram.py -q`.
