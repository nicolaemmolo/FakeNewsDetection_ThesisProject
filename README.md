Fake News Detection – Project for Thesis

# Continual Fake News Detection

## Overview
This repository contains the code for the Master's Thesis: **"Evaluating the Robustness of Fake News Detection Models under Semantic and Temporal Concept Drift"** by Nicola Emmolo (University of Pisa).

Fake News Detection (FND) is often treated as a static task. However, the real-world news ecosystem is highly dynamic, characterized by rapid changes in topics and vocabulary (Concept Drift). When state-of-the-art models are trained on static datasets, they suffer from **Catastrophic Forgetting**, failing to retain knowledge of past events. 

This project addresses this issue by evaluating content-based FND models in a **Continual Learning (CL)** setting. It analyzes the behavior of various architectures—from traditional Machine Learning to Deep Learning and state-of-the-art Transformers—ensuring they can continuously learn from new data without degrading performance on older tasks.

## Project Structure
The repository is organized as follows:

* **`data/`**: Contains the datasets in JSON format, divided by source (e.g., GossipCop, PolitiFact) and labeled as legitimate or fake news.
* **`scripts/`**: Executable scripts to run the different phases of the pipeline:
    * `download_data.py`: Downloads the required datasets.
    * `data_preprocessing.py`: Cleans and prepares the text data for the models.
    * `train.py`: Trains the FND models using various Continual Learning strategies.
    * `evaluate.py`: Tests the models and calculates performance metrics.
* **`src/`**: The core source code of the project:
    * `continual_learning/`: Implementation of Continual Learning strategies (e.g., EWC, Replay).
    * `data/`: Custom dataset classes and data loaders (`dataset.py`).
    * `models/`: Architectures for the FND task, including ML baselines and Transformers.
    * `utils/`: Helper functions and metric calculations.

## Installation

To run this project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone <your-repository-url>
   cd <your-repository-folder>
