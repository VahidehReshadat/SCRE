# SCRE : Special Cargo Relation Extractor using a hierarchical attention-based multi-task architecture 

SCRE is a relation representation model proposed using a hierarchical attention-based multi-task architecture that achieves reasonable performance with limited domain-specific training data. 

The code is part of our paper ["Relation Representation Learning for Special Cargo Ontology"](https://scholar.google.com/citations?user=DNUz3o4AAAAJ&hl=en&authuser=1) that we proposed an ontology population pipeline for the special cargo domain, and as part of the ontology population task, we investigated how to build an efficient information extraction model from low-resource domains based on available domain data in the special cargo transportation domain. For this purpose, a model is designed for extracting and classifying instances of different relation types between each concept pair.

## Overview
A PyTorch implementation of the Bert-base relation extractor for special cargo domain 

## Requirements
Requirements: Python (3.6+), PyTorch (1.2.0+), Spacy (2.1.8+)  

Pre-trained BERT model of [HuggingFace.co](https://huggingface.co)   
Code structure adopted from:
[anago](https://github.com/Hironsan/anago)



## Methodology

I developed a novel hierarchical model with a combination of three different attention-based Natural Language Processing (NLP) models that embeds simple tasks in the low levels of the hierarchy and more complex tasks in the high-level of the hierarchy. The architecture of the attention-based hierarchical multi-task relation representation model for multi-class relation classification is shown in the figure below. The hierarchical multi-task architecture is trained using domain-specific data and used as a based model and feature extractor for the multi-class classifier.

<p align="center">
<img src="https://github.com/VahidehReshadat/SCRE/blob/master/images/Presentation2-2-2.png" alt="overview of HTML" width="400"/>
</p>

**Dataset

data: The training set was created automatically (for more information see ["Relation Representation Learning for Special Cargo Ontology"](https://scholar.google.com/citations?user=DNUz3o4AAAAJ&hl=en&authuser=1)). All datasets for Name Entity Recognition (NER)/Entity Extraction (EE) and Relation Extraction (RE) in CONLL format. 


This repository contains the following folders:

* data/kargo: all datasets for NER/EE/RE in CONLL format. Multi-task modeling as proposed by [Bekoulis et al. (2018)](https://github.com/bekou/multihead_joint_entity_relation_extraction).
* train: training sets with `not_terms_only`: dataset contains all sentences, including sentences without entities (for EE) and `terms_only`: dataset contains only sentences with at least one entity (for EE)
* dev_rel, test_rel: development and test set 1
* online_rel: test set 2 (online documents, based on HTML/PDF excerpts)
* crf: CRF layer implementation for Keras, based on [keras-contrib](https://github.com/keras-team/keras-contrib)
* models: model structure and wrapper for simplified Hiearchical Multi-task Learning from [hmtl](https://github.com/huggingface/hmtl)
* preprocessing.py: preprocessing pipeline for sequential deep learning model
* trainer: training routine for KArgen model, including callbacks.
* main: example of KArgen training and evaluation routine, including saving/loading models.
* infer: example of extraction with the trained models, visualization with [displaCy](https://explosion.ai/demos/displacy)
