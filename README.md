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
- [Taks](#tasks)
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
Our main objective is to better understanding customer preferences through their purchases.
We see a customer as a decision function when he comes to the supermarket. Once facing the shelf of a type of product he wants to buy, the customer assesses what are the different alternatives available. Considering the size, price, brand, packaging, and any other information, the customer ranks all the products in his mind and finally chooses his preferred alternative.
We have at our disposal a list of P expressed preferences. These preferences illustrate that a customer has preferred - or chosen - X[i] over Y[i]. We split the task of determining customers preferences into two sub-tasks:
    -  We want to clusterize the customers through their purchases so that customers with similar decisions are grouped together.
    - We want to determine for each cluster the decision function that lets the customers rank all the products.

## Tasks
You are asked to:
  - Write a Mixed-Integer Progamming model that would solve both the clustering and learning of a UTA model on each cluster
  - Code this MIP inside the TwoClusterMIP class in python/model.py. It should work on the dataset_4 dataset.
  - Explain and code a heuristic model that can work on the dataset_10 dataset. It should be done inside the HeuristicModel class.

## Deliverables
You will present your results during an oral presentation organized the on Tuesday $13^{th}$ (from 1.30 pm) of February. Exact time will be communicated later. Along the presentation, we are waiting for:

-  A report summarizing you results as well as your thought process or even non-working models if you consider it to be interesting.
-  Your solution of the first assignement should be clearly written in this report. For clarity, you should clearly state variables, constraints and objective of the MIP.
-  A well organized git repository with all the Python code of the presented results. A GitHub fork of the repository is preferred. Add some documentation directly in the code or in the report for better understanding. The code must be easily run for testing purposes.
- In particular the repository should contain your solutions in the class TwoClustersMIP and HeuristicModel in the models.pu file.  If you use additional libraries, add them inside the config/env.ymlfile. The command 'python evaluation.py' will be used to check your models, be sure that it works and that your code complies with it. The dataset used will be a new one, with the same standards as 'dataset\_4' and 'dataset\_10'.

## Resources
- [Gurobi](https://www.gurobi.com/)
- [Example Jupyter Notebook](notebooks/example.ipynb)
- [UTA model](https://www.sciencedirect.com/science/article/abs/pii/0377221782901552)
