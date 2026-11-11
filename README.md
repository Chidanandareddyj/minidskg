# Mini-DSKG

## 1. Description

Mini-DSKG is a knowledge graph.
A knowledge graph is a graph that contains meaning.
The nodes are entities.
The edges are relations.
The nodes and the edges have attributes.

This project connects ML datasets to scientific papers.
A paper connects to a dataset if the title or the abstract includes the dataset name.

The project uses the method of the Data Set Knowledge Graph (DSKG).
The project uses the NetworkX library from the DigitalSreeni tutorial.

## 2. Related documents

(a) Tutorial: Knowledge Graphs in Python Using NetworkX

- Video: https://www.youtube.com/watch?v=n7BTWc2C1Eg
- Code: https://github.com/bnsreenu/python_for_microscopists/blob/master/354_knowledge_graphs_using_NetworkX.ipynb

The tutorial shows how to make a graph in NetworkX.
The tutorial adds attributes to nodes and edges.
The tutorial calculates centrality, communities, and similarity.
The tutorial writes the graph to GraphML.
This project uses those procedures on datasets.

(b) Data Set Knowledge Graph (DSKG)

- Code: https://github.com/michaelfaerber/data-set-knowledge-graph
- Paper: https://direct.mit.edu/qss/article/2/4/1324/108051/The-data-set-knowledge-graph-Creating-a-linked

DSKG finds dataset names in paper abstracts.
DSKG connects each dataset to the papers that include the name.
DSKG then calculates the impact of each dataset.

Mini-DSKG does the same work at a small scale.
Mini-DSKG does not use MAG dumps.
Mini-DSKG does not use GraphDB.
Mini-DSKG does not use SPARQL CONSTRUCT.
Mini-DSKG does not do LDA author disambiguation.

## 3. Graph structure

The graph has four node types:

- Dataset
- Paper
- Author
- Task

The graph has four edge types:

- Paper `--mentioned-->` Dataset
- Paper `--writtenBy-->` Author
- Dataset `--usedFor-->` Task
- Dataset `--similarTo-->` Dataset

The similarTo edge uses the Jaccard score of shared papers.

## 4. Procedure to build the graph

The data flow is:

1. Read the catalog in `data/datasets.csv`.
2. Get papers from OpenAlex.
3. If OpenAlex does not accept the request, get papers from arXiv.
4. Find dataset names in the paper text.
5. Make the NetworkX graph.
6. Write GraphML, analysis, and the HTML graph.

Do the steps below from the project folder.

### 4.1 Install the packages

```bash
python -m pip install -r requirements.txt
```

### 4.2 Get the papers

```bash
python src/fetch_papers.py
```

This command writes `data/papers.jsonl`.

NOTE: You can set `OPENALEX_MAILTO` to your e-mail.
OpenAlex can then give better access.

### 4.3 Find the dataset names

```bash
python src/match_mentions.py
```

This command writes `data/mentions.csv`.
The procedure ignores common English words.
The word `data` is not a dataset name.

### 4.4 Make the graph

```bash
python src/build_graph.py
```

This command writes `data/mini_dskg.graphml`.

### 4.5 Analyze the graph

```bash
python src/analyze.py
```

This command writes `outputs/analysis.json`.

### 4.6 Make the HTML graph

```bash
python src/visualize.py
```

This command writes `outputs/explorer.html`.

### 4.7 Do the check

```bash
python tests/test_match.py
```

The check must pass.
The check must find CIFAR-10 in a sample abstract.
The check must not find the word `data` as a dataset.

## 5. How to examine the graph

You can open these files:

- `notebooks/mini_dskg.ipynb`: the demonstration notebook
- `outputs/explorer.html`: the graph in a browser
- `outputs/analysis.json`: the numbers
- `Mini-DSKG_Presentation_v2.pptx`: the slides

You can ask these questions:

- Which datasets occur in the most papers?
- What is the h-index of a dataset?
- Which datasets connect different groups (betweenness)?
- Which datasets are similar to ImageNet?
- Which datasets apply to a problem text?

The h-index of a dataset is h if h papers each have a minimum of h citations.

## 6. Results of this build

This folder contains a complete graph.

| Item | Count |
| --- | --- |
| Datasets | 78 |
| Papers | 1667 |
| Mentions | 2104 |

ImageNet occurs in 106 papers.
The h-index of ImageNet is 50.

The datasets that are most similar to ImageNet are:

1. Tiny ImageNet
2. CIFAR-10

If the problem text is "object detection in autonomous driving", the graph puts these datasets first:

1. Waymo Open Dataset
2. nuScenes
3. KITTI

## 7. Team

- Name: Chidananda Reddy Jonnala
- Roll number: 23BCB7264
- Phone: +919885228387
- Email: chidanandareddyjonnala@gmail.com
