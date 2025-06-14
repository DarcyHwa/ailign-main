# AIlign - AI-Based Bilingual aligner

AIlign est un script python permettant l'alignement de textes parallèles. 

## 1. Présentation
AIlign s'appuie sur les toutes récentes avancées en apprentissage profond, notamment la possibilité de représenter les phrases par des vecteurs dans des espaces multilingues, comme dans les modèles de Laser (Artetxe & Schwenk, 2018, https://arxiv.org/abs/1812.10464) et Labse (Feng et al. 2022, https://arxiv.org/abs/2007.01852).


Il fonctionne en deux temps :

1. une première étape de **préalignement** permet d'extraire des points d'ancrage. Ceux-ci sont obtenus sur la base de divers indice :
  * en l'absence d'embeddings de phrase, on peut s'appuyer sur les mots contenant des ngrams identiques.
  * si on dispose d'embeddings du type Laser (Facebook AI) ou Labse (Google), on peut calculer une mesure de similarité entre phrases de langues différentes
Lorsque deux phrases sont jugées suffisamment similaire (forte densité de ngrams communs ou cosinus des embeddings dépassant un certain seuil) leur appariement produit un point.
Les points ainsi produits sont ensuite filtrés en fonction de divers critères géométriques (densité locale des points, déviation par rapport à la diagonale, etc.) afin de fournir des points d'ancrage de confiance pour la phase 2.
On obtient ensuite des points qui permettent de définir des intervalles de confiance et de guider l'aligment, comme sur la figure ci-dessous.

![alt text](https://gricad-gitlab.univ-grenoble-alpes.fr/kraifo/ailign/-/blob/main/img/test.fr-de.png)

2. la deuxième étape met en oeuvre un **algorithme de Viterbi** de type **DTW** (Dynamic Time Warping) pour calculer récursivement quel est le meilleur chemin menant à un point (i,j). Un chemin est une succession d'appariements. Les appariements pris en compte par AIlign sont les suivants: 1-1, 0-1, 1-0, 1-2, 2-1, 1-3, 3-1, 1-4, 4-1.

On calcule ainsi tous les chemins possibles entre les points d'ancrage fournis à l'étape précédente, en autorisant une certaine marge autour de ceux-ci (paramètre `dtw_margin`).

Le cout d'un chemin est calculé comme la somme des distances des phrases ou groupes de phrases appariées, en utilisant les plongements de phrases (une distance alternative s'appuyant sur les longueurs de phrases et les probabilités de transition, comme chez Gale & Church 2012, pourrait être facilement implémentée).
Pour un appariement vide (1-0 ou 0-1) on définit une distance constante (paramètre `dist_null`).

3. optionnellement, on peut lancer l'alignement lexical lors d'une troisième étape.

## 2. Utilisation

AIlign fonctionne en ligne de commande. Il peut s'appliquer sur 2 fichiers, ou sur tout un répertoire. Il peut également prendre en entrée une liste de paires de fichiers.

### 2.1. Fonctionnement avec deux fichiers

La ligne de commande prend la forme suivante: 

``` 
python3 ailign.py --inputFile1 FILENAME1 --inputFile2 FILENAME2 --inputFormat INPUTFORMAT -- outputFormats OUTPUTFORMATS --outputDir OUTPUTDIR [--outputFile OUTPUTFILENAME] [--l1 LANG1 --l2 LANG2]
```

Pour lancer optionnellement la troisième étape, on peut rajouter les options :

`--wordAlignment` : alignement mot à mot
`--chunkAlignment` : alignement de chunks (groupes syntaxiques minimaux construits à partir des sorties de Stanza)


Par exemple, avec les données de test présentes dans le répertoire ./test, pour aligner l'original de KHM53 (1857) avec la traduction de Martin (1846), on peut écrire :

```
python3 ailign.py --inputFile1 "test/KHM53.1857.Grimm.de.xml" --inputFile2 "test/KHM53.1846.Martin.fr.xml" --inputFormat xml --outputFormats txt tmx ces xml --outputFile "test/KHM53" --addAnchor 
```
Des balises <anchor> seront ajoutées dans le xml de sorties afin d'aligner le fichier 2 avec le fichier 1.

Nota Bene : OUPUTFILENAME ne doit pas contenir d'extension : celle-ci est ajoutée en fonction des formats de sortie. 

`--xmlGuide` : indique quelles sont les balises à prendre en compte pour la définition des segments (p.ex <s> <l> <seg> etc.)
`--addAnchor` : permet d'ajouter des balises anchor dans la sortie XML. Dans le cas où `--direction 1->2` est indiqué, les correspondances des ancres du fichier 2 pointent vers le fichier 1, considéré comme pivot. Si on a `--direction 1<->2`, les correspondances des ancres sont ajoutées dans les deux directions.


Si le format de fichier OUTPUTFILENAME n'est pas indiqué, le nommage du fichier sera effectué automatiquement, à partir de FILENAME1, FILENAME2, LANG1, LANG2 et l'extension du format de fichier.

Les formats reconnus en entrée sont les suivants :
- `txt` : format texte brut. Chaque ligne forme un segment.
- `ces` : format cesAna. Format XML contenant les segments entre balises "<s>"
- `arc` : format Arcade.
- `tsv` : format TSV (dans ce cas spécifier les paramètres `--col1` et `--col2` pour indiquer les indices de colonnes contenant l1 et l2)
- `xml` : format XML (p.ex. XML TEI) contenant des balises de segmentation (typiquement `<s> </s>` mais d'autres balises sont possibles, à définir avec `--xmlGuide`). Le paramètre `--anchorTag` permet d'indiquer les balises correspondant à des zones préalignées (p.ex. des paragraphes, des strophes, des sections, etc.).
- `xml-conll` : format XML contenant entre balise `<s> </s>` des phrases analysées en conll.

Avec les paramètres `--splitSent1` et `--splitSent2` les segments lus en l1 (et respectivement l2) sont ensuite découpés en phrases plus petites. La grammaire de segmentation peut-être adaptée avec la regex donnée par l'argument `--splitSentRegex`. Par défaut, la regex employée est : `(?<=[?;:.!"»…]) (?=[A-Z])`.

Les formats de sortie sont les suivants :
- `cesAlign` : format xml contenant les appariements entre identifiants
- `tmx` : format xml contenant les phrases sources et cibles alignés
- `txt` : format texte brut où les phrases sources et cibles sont séparées par des retours chariots
- `txt2` : format texte brut aboutissant à deux fichiers parallèles où les lignes se correspondent
- `tsv` : format texte brut où les phrases sources et cibles sont séparées par des tabulations et regroupées sur une même ligne (format pour tableur)

Plusieurs formats de sortie peuvent être spécifiés en même temps.

Par défaut les deux phases sont exécutées. Pour lancer seulement la phase 1 (préalignement), rajouter le paramètres `--doNotRunDTW`.

Autres paramètres contrôlant la sortie :
- `--writeAnchorPoints` : écrit le fichier `.anchor` contenant les points d'ancrage
- `--l1` et `--l2` : indiquent les langues concernées (p.ex. `fr` et `en`), utiles pour la sortie `tmx`.
- `--savePlot` : enregistre le graphique des points d'ancrage dans un fichier png
- `--showPlot` : affiche le graphique des points d'ancrage.
- `--verbose` : affiche les traces d'exécution
- `--veryVerbose` : affiche plus de traces d'exécution

### 2.2. Fonctionnement avec plusieurs fichiers dans des répertoires

On place les fichiers sources et cibles dans un répertoire. Pour l'identification des langues de chaque fichier, et des correspondances entre fichiers, il est important de nommer les fichiers suivant le schéma : NAME.xxx.LANG.EXT (p.ex. `blancheNeige.1871.fr.txt`). 
La ligne de commande prend la forme suivante: 

``` 
python3 ailign.py --inputDir INPUTDIR [--filePattern FILEPATTERN] --inputFormat INPUTFORMAT -- outputFormats OUTPUTFORMATS --outputDir OUTPUTDIR --l1 LANG1 [--l2 LANG2]
```
- Le paramètre `filePattern` est fixé par défaut à `"(.*)[.](\w\w\w?)[.]\w+$"`. Il s'agit d'une expression régulière contenant deux parenthèses capturantes. La première capture la partie commune des fichiers à aligner et la seconde capture la langue sur deux ou trois caractères. Ce paramètre peut nécessiter une adaptation en cas de schéma de nommage plus compliqué. Par exemple, supposons que l'on gère plusieurs traductions d'une même oeuvre comme ci-dessous :

```
KHM53.1819.grimm.de.txt
KHM53.1846.martin.fr.txt
KHM53.1869.alsleben.fr.txt
```

Dans ce cas la partie commune doit être définie comme la partie qui précède le premier point. Le pattern sera donc :
`"(.*?)[.].*[.](\w\w\w?)[.]\w+$"` 

- le paramètre `l1` est également obligatoire, il définit la langue source : ici on choisirait par exemple `--l1 de`

- le paramètre `l2` vaut `"*"` par défaut. Dans ce cas, tout fichier contenant la même partie commune qu'un fichier de l1, et respectant le pattern filePattern, sera aligné avec ce fichier de l1, quelle que soit sa langue.

- pour le nommage des fichiers cibles, celui-ci sera obtenu avec la concaténation des deux noms de fichier alignés.

### 2.3. Fonctionnement avec un fichier list.txt contenant la liste des paires de fichiers à traiter

La liste des paires de fichiers peut être spécifiée dans un fichier, chaque nom de fichier étant séparé par une tabulation. Le format est :

``` 
python3 ailign.py --inputFileList FILENAMELIST --inputFormat INPUTFORMAT --outputFormats OUTPUTFORMATS --outputDir OUTPUTDIR
```

Par exemple, avec les données de test présentes dans le répertoire ./test, pour aligner toutes les traductions de l'Iliade avec la traduction de Certon comme pivot, on peut écrire :

```
python3 ailign.py --inputFileList "test/filelist.txt" --inputFormat xml --outputFormats txt tmx ces xml --outputDir "test/aligned" --addAnchor --xmlGuide l --direction "1->2" --anchorTag "lg"
```

Les balises <lg> font office de balises de préalignement (anchorTag).

## 3. Paramétrages

### 3.1 Extraction des points d'ancrage

Les principaux paramètres sont :

- `--labseThreshold` (défaut=0.6) : la similarité cosinus minimum avec Labse pour retenir un point d'ancrage.
- `--kBest` (défaut=3): nombre de meilleurs points à enregistrer pour chaque ligne I et colonne J
- `--margin` (défaut=0.05): marge entre le meilleur point d'une ligne (ou d'une colonne) et ses concurrents. Si la différence entre les scores est inférieure à margin, le point n'est pas retenu.
- `--minDensityRatio` (défaut=0.5) : rapport entre densité locale et densité moyenne des points pour conserver un point d'ancrage
- `--deltaX` (défaut=3): longueur horizontale du couloir diagonal autour du point pour le calcul de la densité
- `--deltaY` (défaut=2): largeur verticale du couloir diagonal autour du point pour le calcul de la densité


### 3.2 Extraction de l'alignement complet

Les principaux paramètres sont :
- `--dtwMargin` (défaut=3) : l'écart maximum entre le chemin calculé et un point d'ancrage (verticalement ou horizontalement).
- `--distNull` (défaut=0.8): la distance correspondant à un appariement vide (1-0 ou 0-1)
- `--penalty_2_2` (défaut=0.1): pénalité de granularité large liée à un appariement 2-2 (pour favoriser 2 appariements 1-1 correspondant).
- `--penalty_n_n` (défaut=0.1): pénalité de granularité large liée à un appariement 1-n ou n-1.

## 4. Référence à citer

Kraif, Olivier (2024). Adaptative Bilingual Aligning Using Multilingual Sentence Embedding. Pre-print Arxiv. https://arxiv.org/abs/2403.11921
Kraif, Olivier (2025). Alignement bi-textuel adaptatif basé sur des plongements multilingues. Actes de Coria-TALN 2025, Marseille, 1-5 juillet 2025.

## 5. Modules python requis

BTrees
lxml
matplotlib
numpy
sentence_transformers
stanza

Installation : 

`pip3 install BTrees lxml matplotlib numpy sentence_transformers stanza`


## Crédits et licence

Conçu et réalisé par Olivier Kraif, Université Grenoble Alpes, 2023-2025. Pour la partie alignement lexical / alignement des chunks, conçu et réalisé avec Elnaz Jalilian.

Merci à Inès Adjoudj, Beliz Ozkan et Beining Yang pour leurs contributions lors de leur stage.

(cc) CC-BY-NC
