# Pediatric Recommendation System - AI Course Capstone Project

This repository contains the final report and supporting materials for our AI Course Capstone Project: a Pediatric Recommendation System.  The system uses AI to provide parents with accessible medical information (diets, precautions, medications) and assist medical professionals with faster, more accurate diagnoses.  It leverages NLP and a knowledge graph for personalized recommendations and insights.

**Note:** This project was developed as part of the Samsung Innovation Campus curriculum and is copyrighted by Copyright© 1995-2025 Samsungq.  Reproduction or redistribution without written consent is prohibited. Instructor review is required.

## Table of Contents

1. [Introduction](#introduction)
    1.1 [Background Information](#background-information)
    1.2 [Motivation and Objective](#motivation-and-objective)
    1.3 [Members and Role Assignments](#members-and-role-assignments)
    1.4 [Schedule and Milestones](#schedule-and-milestones)
2. [Project Execution](#project-execution)
    2.1 [Data Acquisition](#data-acquisition)
    2.2 [Training Methodology](#training-methodology)
    2.3 [Workflow](#workflow)
    2.4 [System Diagram](#system-diagram)
3. [Results](#results)
    3.1 [Data Preprocessing](#data-preprocessing)
    3.2 [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
    3.3 [Modeling](#modeling)
    3.4 [User Interface and Role-Based AI Interaction](#user-interface-and-role-based-ai-interaction)
    3.5 [Testing and Improvements](#testing-and-improvements)
4. [Projected Impact](#projected-impact)
    4.1 [Accomplishments and Benefits](#accomplishments-and-benefits)
    4.2 [Future Improvements](#future-improvements)
5. [Team Member Review and Comment](#team-member-review-and-comment)
6. [Instructor Review and Comment](#instructor-review-and-comment)

## Introduction

### 1.1 Background Information

Artificial intelligence (AI) is transforming pediatric healthcare by offering accurate, efficient, and tailored insights, allowing both parents and doctors to make educated decisions regarding children's health. Current automated systems frequently suffer with insufficient pediatric specific datasets, contextual comprehension, and accessibility concerns. To solve these issues, we suggest an AI-powered recommendation system that may provide parents with medical information like (Diets, precautions, Medications and can also predict the disease from provided symptoms) and help medical professionals make faster, more accurate diagnoses. This system uses powerful natural language processing models to steer discussions, forecast illnesses, and provide personalized suggestions, closing major gaps in pediatric healthcare and enhancing decision-making for both caretakers and professionals.

### 1.2 Motivation and Objective

The idea for this initiative stems from the urgent need for accessible and credible medical advice, particularly in pediatric healthcare. Many parents struggle to grasp their child's symptoms and determine suitable next measures, while Pediatricians encounter time limits and a high demand for precise diagnosis. By integrating AI, we want to improve healthcare outcomes by delivering trustworthy advice and a user-friendly solution that helps parents get their needed information and helps doctors diagnose diseases promptly in a faster way using a graph knowledge system that can give them what they need from pediatric books and with the reference of course. The goal of this project is to create a powerful Pediatric knowledge system that is integrated with predictive models, recommendation systems, and an AI chatbot model providing individualized medical insights and practical recommendations to close the gap between healthcare demands and successful treatments.

### 1.3 Schedule and Milestones

**Project Initiation and Planning**
* Start Date: December 16, 2024
* End Date: December 18, 2024
    * Defining project goals and objectives.
    * Assigning roles and responsibilities.
    * Developing a roadmap for execution.

**Data Acquisition and Preparation**
* Start Date: December 19, 2024
* End Date: December 26, 2024
    * Gather pediatric healthcare datasets (symptoms, diseases, recommendations).
    * Clean and preprocess the data for modeling.
    * Augment and normalize the dataset for better diversity.

**Model Development**
* Start Date: December 27, 2024
* End Date: January 7, 2025
    * Ontology Mapping and preparation using OWL and RDFLib from the csv collected Dataset.
    * Train Random Forest model for disease prediction based on symptoms.
    * Train NER Models BioBERT and ClinicalBERT and test both of them.
    * Use the chosen NER Model for extracting well annotations and entities from PDFs dataset and prepare a json File.

**System Integration**
* Start Date: January 8, 2025
* End Date: January 15, 2025
    * Enrich the Ontology with the prepared json files and map it in both directions.
    * Train Sentence-BERT model for designing the recommendation system.
    * Develop TF-IDF vectorizer for recommendation ranking.
    * Use the LLM Model for personalized medical resource recommendations.
    * Test the recommendation system for Diets, precautions and medications from symptoms and diseases. Test the LLM model for extracting and resuming a good information from books.
    * Test the enriched Ontology using Protégé.

**Validation**
* Start Date: January 16, 2025
* End Date: January 22, 2025
    * Integrate the trained models saved as PKL files.
    * Validate the accuracy of disease prediction and recommendations.
    * Fine-tuning Bio-clinicalBERT Model on our datasets for smoother text generation.

**Deployment and Feedback**
* Start Date: January 23, 2025
* End Date: January 26, 2025
    * Prepare and deploy the chatbot for limited testing.
    * Gather feedback for iterative improvements.
    * Prepare the final presentation and the final report.

**Deployment and Feedback (Continued)**
* Start Date: January 27, 2025
* End Date: January 29, 2025
    * Go back and correct specific faults.
    * Fine-tune a Qwen-1.5B Deepseek Model for our specific goals.
    * Compare the chatbot results between the Bio-clinicalBERT Model and the Deepseek.
    * Thinking about adding some new goals and discussing new objectives for the future.


## Project Execution

### 2.1 Data Acquisition

Our work showcases a comprehensive pipeline for cleaning, augmenting, annotating, and categorizing textual medical data, which is integral to building a reliable pediatric recommendation and disease prediction system. Here's a breakdown of the steps:

**Text Cleaning and Synonym Mapping**
The `clean_text` function processes raw textual data by removing unnecessary metadata, numeric entries, and special characters. It replaces medical abbreviations or colloquial terms with their standardized synonyms (e.g., "flu" becomes "influenza") using a predefined dictionary. Additionally, custom stop words and punctuation are filtered to retain only meaningful content.

**File-Level Text Cleaning**
The `process_txt_files` function applies `clean_text` to files in a directory, ensuring each file is processed systematically and stored in a specified output directory. This step standardizes medical datasets, eliminating noise and ambiguity.

**Disease-Symptom Dataset Normalization**
In another phase, `clean_disease_symptom_dataset` focuses on cleaning structured datasets by:
* Removing unnecessary columns like Age and Gender.
* Mapping binary values ("Yes", "No") to numeric (1, 0).
* Standardizing disease names using a mapping dictionary. The cleaned dataset ensures consistency and readiness for downstream machine learning models.

**Text Normalization and Augmentation**
To enhance the robustness of the system, the `normalize_text` function standardizes terms like "mg" to "milligrams." Following this, `augment_text` introduces diversity by replacing words with synonyms derived from the WordNet lexical database. This augmentation expands the dataset, aiding in model generalization.

**Text Annotation with Spa
