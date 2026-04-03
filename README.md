# Fake News Detection – Project for MD Thesis

## Overview
This repository contains the code for the Master's Thesis: **"Evaluating the Robustness of Fake News Detection Models under Semantic and Temporal Concept Drift"** by Nicola Emmolo (University of Pisa).

Fake News Detection (FND) is often treated as a static task. However, the real-world news ecosystem is highly dynamic, characterized by rapid changes in topics and vocabulary (Concept Drift). When state-of-the-art models are trained on static datasets, they suffer from **Catastrophic Forgetting**, failing to retain knowledge of past events. 

This project addresses this issue by evaluating content-based FND models in a **Continual Learning (CL)** setting. It analyzes the behavior of various architectures, from traditional Machine Learning to Deep Learning and state-of-the-art Transformers, ensuring they can continuously learn from new data without degrading performance on older tasks.

## Project Structure
The repository is organized as follows: