# Fake News Detection – Project for MD Thesis

## Overview
This repository contains the code for the Master's Thesis: **"Evaluating the Robustness of Fake News Detection Models under Semantic and Temporal Concept Drift"** by Nicola Emmolo (University of Pisa).

Fake News Detection (FND) is often treated as a static task. However, the real-world news ecosystem is highly dynamic, characterized by rapid changes in topics and vocabulary (Concept Drift). When state-of-the-art models are trained on static datasets, they suffer from **Catastrophic Forgetting**, failing to retain knowledge of past events. 

This project addresses this issue by evaluating content-based FND models in a **Continual Learning (CL)** setting. It analyzes the behavior of various architectures, from traditional Machine Learning to Deep Learning and state-of-the-art Transformers, ensuring they can continuously learn from new data without degrading performance on older tasks.


## Experimental Setup
Here is a brief recap of the evaluation protocols, models, and strategies used in this study to handle Concept Drift in Fake News Detection.

**Evaluation Scenarios**
The datasets were organized into two distinct sequences to simulate different real-world drift conditions:
* **Topic-Incremental Scenario:** A sequence of 7 tasks (Politics, General, Covid, Syria, Islam, NotreDame, Gossip) representing semantic shifts between different news domains.
* **Time-Incremental Scenario:** A chronological sequence of 5 tasks (2011-15, 2016, 2017, 2019, 2020) representing the temporal evolution of news to test the impact of time disjointedness.

**Models Evaluated**
The experiments compare 8 different architectures across three main families:
* **Machine Learning:** Linear SGD, Passive-Aggressive (PA), and Naive Bayes (NB).
* **Deep Learning from Scratch:** CNN and BiLSTM (initialized with Word2Vec embeddings).
* **Pre-trained Transformers:** BERT, RoBERTa, and DeBERTa (base versions).

**Continual Learning (CL) Strategies**
To mitigate Catastrophic Forgetting, the following strategies were implemented:
* **Experience Replay:** Stores a subset of past samples in a memory buffer and interleaves them with new data during training.
* **Learning without Forgetting (LwF):** Uses knowledge distillation, acting as a "teacher" to help the new model ("student") remember past task structures without storing old data.
* **Elastic Weight Consolidation (EWC):** A regularization technique that protects the most critical neural network weights of past tasks from being drastically changed.
* **Hybrid (LwF + Replay):** Combines an explicit memory buffer with knowledge distillation for a balanced approach.

**Evaluation Bounds**
To properly measure the success of the CL strategies, two baselines were established:
* **Lower Bound (Sequential Fine-tuning):** The model trains on new tasks sequentially with no memory countermeasures (showing the maximum Catastrophic Forgetting).
* **Upper Bound (Offline / Joint Training):** The ideal scenario where the model trains on all data simultaneously (violating CL constraints, serving as the maximum accuracy ceiling).

---

## Project Structure

Based on the experimental setup, the repository is organized as follows:

### `datasets/`
This folder contains the data collected for the experiments.
*Note: Due to file size limitations, the **ISOT** and **Kaggle-Clement** datasets are not included in this repository.*

The datasets utilized cover various domains and timeframes:
* **Celebrity:** Focuses on the entertainment and gossip domain, pairing fake tabloid news with contemporary legitimate news.
* **CIDII:** Dedicated to the Islamic religious domain, covering sub-themes like women’s rights and inter-religious relations.
* **Fa-KES:** Centers on the Syrian war conflict, focusing on articles that distort factual war information.
* **FakeVsSatire:** Designed specifically to distinguish between malicious fake news and political satire in the US.
* **Horne:** Focuses on US political "hard" news, primarily surrounding the 2016 presidential elections.
* **Infodemic:** A dataset dedicated to the COVID-19 health crisis, verifying information against official medical handles and fact-checkers.
* **ISOT (Not in repo):** Focuses heavily on the 2016 US Presidential election.
* **Kaggle-Clement (Not in repo):** Covers Politics and World News during the 2016-2017 period.
* **Kaggle-Meg:** Covers multiple themes including politics, health, and conspiracy theories.
* **LIAR-PLUS:** Contains short political statements accompanied by extracted expert justifications from PolitiFact.
* **Politifact:** Derived from fact-checking activities regarding US politics, composed of short statements from speeches and interviews.
* **NDF:** Focused specifically on the Notre Dame Cathedral fire of April 2019, including a mix of tweets and articles.

### `src/`
This directory contains the core execution files of the project. Inside, you will find:
* Scripts for hyperparameter searching.
* The main execution files to run and test the various models.
* Directories containing the output results of the experiments.
* Jupyter Notebooks used for generating plots and visualizing the data.

### `Word2Vec_GoogleNews300/`
This folder contains the pre-trained **Word2Vec embeddings (Google News, 300d)** used to initialize the Deep Learning models (CNN and BiLSTM). 
*Note: Due to its significant file size, this directory is not included in the repository. You can download it on [https://code.google.com/archive/p/word2vec/](https://code.google.com/archive/p/word2vec/).*