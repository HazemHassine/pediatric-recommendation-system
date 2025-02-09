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

**Text Annotation with SpaCy**
The `annotate_text` function leverages SpaCy's Named Entity Recognition (NER) to extract and label entities like symptoms, treatments, and medical conditions from the cleaned data. These annotations help understand the context and importance of the extracted text segments.

**Text Categorization**
The `categorize_text` function organizes text into categories (e.g., "symptoms," "treatment," "diagnosis") using regular expression patterns. This categorization simplifies data interpretation and prepares the information for tailored recommendation systems.

**Pipeline Integration**
The `process_files` function ties all steps together. For each file in the cleaned dataset:
* Text is normalized.
* Augmented text is generated.
* Annotations and categorized segments are identified.
* The augmented text is saved for further use.

This multi-step approach ensures the data pipeline processes diverse inputs, enriches datasets with augmented content, and structures the data into meaningful formats for machine learning and AI applications in pediatric healthcare.

### 2.2 Training Methodology

The training methodology for the project encompassed several components, each tailored to meet specific needs. For the Random Forest (RF) model, the training involved preprocessing the dataset by encoding categorical features, normalizing numerical data, and addressing class imbalances. The model was trained on labeled datasets to predict diseases based on symptoms, using cross-validation to ensure generalization. An LLM was used to model disease-symptom relationships by encoding them as text (e.g., "Disease X causes Symptom Y") and querying with prompts. The model was trained using node embeddings to capture semantic connections and improve disease predictions. The recommendation system was developed using a TF-IDF vectorizer to convert medical texts into feature vectors, followed by cosine similarity calculations to rank recommendations. The chatbot leveraged pre-trained transformer models via Hugging Face. Fine-tuning involved training on medical datasets to optimize the chatbot for pediatric healthcare contexts, ensuring it could integrate predictive insights and offer relevant recommendations. All components were tested iteratively to refine their accuracy and usability, resulting in a cohesive system for pediatric healthcare support.

### 2.3 Workflow

1. **Data Preprocessing:** The raw dataset consisted of medical records, disease symptoms, and treatment recommendations in CSV and PDF formats. Preprocessing involved:
    * Data extraction: In this phase, PDFs were converted into .txt files, and irrelevant text like author names, titles, and https links were removed.
    * Data Cleaning: Irrelevant text (e.g., page numbers, metadata), stopwords, and unnecessary characters were removed. Medical synonyms were normalized using a predefined dictionary (e.g., "flu" → "influenza").
    * Tokenization: Sentences were converted into individual tokens for better text representation.
    * Augmentation: Data was enriched by replacing terms with synonyms and categorizing text into "Symptoms," "Treatment," and "Diagnosis" categories using SpaCy and NLTK tools.
    After these steps, the data was tested using visualization techniques like most frequent words, word and character counts per file, and word clouds.

2. **Feature Engineering:**
    * TF-IDF Vectorization: Cleaned text was transformed into numerical representations for machine learning models.
    * Ontology Preparation: An ontology was prepared from the CSV files.
    * Creation of NER Models: Two NER models, BioBERT and ClinicalBERT, were trained on entities extracted from the initial ontology (due to licensing issues with SNOMED CT and UMLS).
    * Entities Extraction from TXT Files: The best NER model (ClinicalBERT) was used to extract entities and annotations to enrich the ontology.
    * Enrich the Ontology: The ontology was enriched with new entities and tested on Protégé.
    * Graph Construction: A knowledge graph was created with nodes representing symptoms, diseases, and treatments, and edges representing relationships like co-occurrence or causal links.
    * Label Encoding: Categorical data was converted into numerical formats.

3. **Model Training:**
    * Random Forest (RF) Model: Trained using preprocessed data to predict diseases based on symptoms. Cross-validation was performed.
    * Large Language Model (LLM): Trained on the constructed graph to capture complex relationships from parts of the books and summarize them.
    * Chatbot Fine-Tuning: The ClinicalGPT model from Hugging Face was fine-tuned.

4. **Integration:** All trained components were integrated into a unified chatbot pipeline, enabling the chatbot to:
    * Predict diseases using the RF model.
    * Recommend treatments via the recommendation system.
    * Engage in conversational AI.

5. **Testing and Validation:**
    * Unit Testing: Each component was tested individually.
    * End-to-End Testing: Real-world scenarios were simulated.
    * Metrics: Accuracy, relevance, and fluency were measured.

### 2.4 System Design

The system design includes the following components: Random Forest (RF) model, Large Language Models (LLM), and a Chatbot Fine-Tuning model.  These components work together in a pipeline to provide disease prediction, recommendations, and conversational interaction.

## Results

### 3.1 Data Preprocessing

Data preprocessing involved multiple steps:
* Cleaning: Irrelevant data was removed. Medical synonyms were normalized.
* Tokenization: Sentences were split into tokens.
* Vectorization: TF-IDF vectorization was used.
* Tools Used: pandas, Scikit-learn, SpaCy, NLTK.

### 3.2 Exploratory Data Analysis (EDA)

EDA revealed:
* Disease Trends: Respiratory infections, fevers, and gastrointestinal issues were most common.
* Symptom Co-occurrence: Correlations were observed between symptoms like fever and fatigue, and cough and difficulty breathing.
* Age and Gender Patterns: Younger children were more prone to viral infections, while older children exhibited symptoms of chronic conditions.
* Visualization Tools: Bar charts and heatmaps were used.

### 3.3 Modeling

The modeling phase included:
* Disease Prediction: A Random Forest model was trained and achieved 68% accuracy. Fine-tuned models like ClinicalGPT were also used.
* Recommendation System: TF-IDF vectorization and cosine similarity were used.
* Chatbot Integration: The chatbot was trained to understand context and generate responses.

### 3.4 User Interface and Role-Based AI Interaction

The user interface is designed to be intuitive and user-centric. A key feature is its role-based system, allowing users to select different categories—such as "Pediatric," "Medical Student," or "Parent"—to receive responses specifically tailored to their needs. The chat interface is simple, with an input field and a "Send" button. A unique functionality allows users to interrupt the AI mid-response. The system is free for all users.

**Differences in AI Responses Based on User Roles:**
* **Parent Role:** Simplified, reassuring, and practical guidance. Example: A parent asking about a dry cough, clogged nostrils, and sore throat received information about common causes like cold, flu, or allergies, with an emphasis on seeing a doctor.
* **Medical Student Role:** Detailed, technical, and research-oriented information. Example: A medical student asking about weak eyesight in children aged 5-11 received an in-depth answer covering refractive errors, genetic and environmental risk factors, and references to textbooks and PubMed search terms.
* **Pediatric Role:** Clinical and diagnostic considerations. Example: A pediatrician inquiring about a 4-year-old female with red spots and button-like skin texture received information about possible conditions like viral exanthems, allergic reactions, or dermatological conditions, with an emphasis on physical examination and diagnostic tests.

### 3.5 Testing and Improvements

**Testing Outcomes:**
The system was tested on diverse scenarios, including edge cases. The RF model achieved 68% accuracy for disease prediction, and the recommendation system demonstrated good relevance.

**Improvements:**
* Context-Aware Responses: ClinicalGPT integration improved the chatbot's ability to understand conversations.
* Error Handling: Robust error handling was implemented.

## Projected Impact

### 4.1 Accomplishments and Benefits

The system offers:
* Improved Accessibility to Pediatric Advice
* Enhanced AI-Driven Medical Assistance
* Support for Pediatricians
* Parent-Friendly Design

### 4.2 Future Improvements

* Expanded Dataset Integration
* Multi-Language Support
* Medical Institution Collaboration
* Real-Time Monitoring
* Interactive Features
* Integration of a predictive model for illness stage detection
* Integration of Google Maps Model for finding pediatricians and pharmacies

