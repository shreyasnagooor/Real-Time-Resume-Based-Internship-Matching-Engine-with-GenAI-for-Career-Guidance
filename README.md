# Job Recommendation System using NLP and Cosine Similarity

Welcome to the **Job Recommendation System**! This project matches job descriptions from internship data with resumes using natural language processing (NLP) techniques. By leveraging cosine similarity, it finds the most relevant job descriptions for given resumes, helping job seekers find their best-fit opportunities.

> **Career Guidance**: For aspiring professionals in AI/ML, explore **[HackML](https://hackml.vercel.app/)** for resources and mentorship in your career journey!

---

## Table of Contents

| Section          | Link                                                                 |
|------------------|----------------------------------------------------------------------|
| **Overview**      | [#overview](#overview)                                               |
| **Technologies**  | [#technologies-used](#technologies-used)                             |
| **Data Preparation** | [#data-preparation](#data-preparation)                           |
| **Preprocessing** | [#preprocessing](#preprocessing)                                     |
| **Model**         | [#model](#model)                                                     |
| **Evaluation**    | [#evaluation](#evaluation)                                           |
| **Visualization** | [#visualization](#visualization)                                     |
| **Installation**  | [#installation](#installation)                                       |
| **Usage**         | [#usage](#usage)                                                     |
| **Contributing**  | [#contributing](#contributing)                                       |
| **License**       | [#license](#license)                                                 |

---

## Overview

This system is designed to recommend job descriptions based on the similarity between the text of resumes and job descriptions. It uses **Sentence Transformers** to convert text into embeddings, then calculates **cosine similarity** to rank job descriptions for each resume. The goal is to help job seekers identify the most relevant opportunities based on their resumes.

---

## Technologies Used

- **Python 3.x**
- **Libraries**:
  - `pandas` – For data manipulation
  - `numpy` – For numerical operations
  - `re` – For regular expressions
  - `nltk` – For text processing and tokenization
  - `sentence-transformers` – For sentence embeddings and semantic similarity
  - `scikit-learn` – For cosine similarity and model evaluation
  - `seaborn` & `matplotlib` – For data visualization

---

## Data Preparation

This project requires two datasets:
1. **Internship Data**: Contains job descriptions for internships.
2. **Resume Data**: Contains resumes to be matched with the job descriptions.

### Required Columns:
- **Internship Data (`df1`)**: `Cleaned_Job_Description`
- **Resume Data (`df2`)**: `Cleaned_Text`

Both datasets should be cleaned and preprocessed before use.

---

## Preprocessing

The text data is cleaned and preprocessed through the following steps:
1. **Cleaning**: Removing non-alphanumeric characters, numbers, and extra spaces.
2. **Tokenization**: Breaking text into individual tokens (words).
3. **Stopwords Removal**: Removing common, non-meaningful words (e.g., "the", "is").
4. **Lemmatization**: Converting words to their root form (e.g., "running" → "run").

Both job descriptions and resumes undergo this preprocessing before being used in the recommendation system.

---

## Model

We use the **Sentence Transformer** model (`all-MiniLM-L6-v2`) to encode both the job descriptions and resumes into vector representations (embeddings). These embeddings are compared using **cosine similarity** to determine the degree of match between each resume and job description.

The system uses a **cosine similarity threshold of 0.7** to classify whether a resume matches a job description.

---

## Evaluation

We evaluate the recommendation system using the following metrics:
- **Precision**: The proportion of relevant job descriptions among the recommended ones.
- **Recall**: The proportion of relevant job descriptions retrieved.
- **F1 Score**: The harmonic mean of precision and recall.
- **Accuracy**: The overall correctness of the model in making recommendations.

---

## Visualization

The distribution of cosine similarity scores is visualized using **seaborn** to observe how well the model differentiates between relevant and irrelevant job descriptions. A threshold line is drawn at `0.7` to separate relevant and irrelevant matches.

---

## Installation

Follow the steps below to set up the project:

### 1. Install required libraries:

```bash
!pip install sentence-transformers
!pip install seaborn
!pip install nltk
