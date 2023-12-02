<p align="center">
  <img aligne="center" src="images/CS.png" width="45%" />
  <img aligne="center" src="images/artefact.png" width="50%" ver />
</p>

# Decision Systems and Preferences
Auriau Vincent,
Belahcene Khaled,
Mousseau Vincent

## Table of Contents
- [Decision Systems and Preferences](#decision-systems-and-preferences)
- [Repository Usage](#repository-usage)
- [Context](#context)
- [Deliverables](#deliverables)
- [Resources](#resources)

## Repository usage
1.  Install [git-lfs ](https://docs.github.com/en/repositories/working-with-files/managing-large-files/installing-git-large-file-storage), it will be needed to download the data
2. Fork the repository, the fork will be used as a deliverable
3. Clone your fork and push your solution model on it

The command 
```bash
conda env create -f config/env.yml
conda activate cs_td
python evaluation.py
``````
will be used for evaluation, with two other test datasets. Make sure that it works well.

## Context
Our objective is to clusterize the decision makers through their observed preferences and to learn the decisions functions of each of these clusters.

## Deliverables
You will present your results during an oral presentation organized the on Tuesday $13^{th}$ (from 1.30 pm) of February. Exact time will be communicated later. Along the presentation, we are waiting for:

-  A report summarizing you results as well as your thought process or even non-working models if you consider it to be interesting.
-  Your solution of the first assignement should be written in a document and coded inside the class TwoClustersMIP of the models.py file. Also, provide a conda environment (as .yaml) if you use any additional libraries. The command 'python evaluation.py' will be used to check your model, be sure that it works and that your code complies with it. The dataset used will be a new one, with the same standards as 'dataset\_4'.
-  A well organized git repository with all the Python code of the presented results. A GitHub fork of the repository is preferred. Add some documentation directly in the code or in the report for better understanding. The code must be easily run for testing purposes.

## Resources
- [Gurobi](https://www.gurobi.com/)
- [Example Jupyter Notebook](notebooks/example.ipynb)
