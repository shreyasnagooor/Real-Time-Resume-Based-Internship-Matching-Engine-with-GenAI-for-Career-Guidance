#  Real-Time Resume-Based Internship Matching Engine with Gen AI for Career Guidance

Welcome to the **Job Recommendation System**! This project matches job descriptions from internship data with resumes using natural language processing (NLP) techniques. By leveraging cosine similarity, it finds the most relevant job descriptions for given resumes, helping job seekers find their best-fit opportunities.

> **Career Guidance**: For aspiring professionals in AI/ML, explore **[HackML](https://hackml.vercel.app/)** for resources and mentorship in your career journey!

---

## Table of Contents

| Section          | 
|------------------| 
| **Overview**      | 
| **Technologies**  | 
| **Data Preparation** | 
| **Preprocessing** | 
| **Model**         | 
| **Evaluation**    | 
| **Visualization** | 
| **Installation**  | 
| **Usage**         | 
| **Contributing**  | 
| **License**       | 

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
2. Download necessary NLTK datasets:
python
Copy
Edit
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
3. Upload your datasets:
Make sure the datasets cleaned_internship_data.csv and cleaned_resumes.csv are uploaded to your environment or local machine.

Usage
1. Load the datasets:
python
Copy
Edit
import pandas as pd

df1 = pd.read_csv('cleaned_internship_data.csv')
df2 = pd.read_csv('cleaned_resumes.csv')
2. Preprocess the text data:
Use the preprocess_text() function to clean the data:

python
Copy
Edit
df1['Cleaned_Job_Description'] = df1['Cleaned_Job_Description'].apply(preprocess_text)
df2['Cleaned_Text'] = df2['Cleaned_Text'].apply(preprocess_text)
3. Encode the text using Sentence Transformers:
python
Copy
Edit
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

job_vectors = model.encode(df1['Cleaned_Job_Description'].dropna().tolist())
resume_vectors = model.encode(df2['Cleaned_Text'].dropna().tolist())
4. Calculate the cosine similarity:
python
Copy
Edit
from sklearn.metrics.pairwise import cosine_similarity

similarity_matrix = cosine_similarity(resume_vectors, job_vectors)
5. Generate recommendations:
python
Copy
Edit
top_n = 5
recommendations = []

for i, resume_vector in enumerate(resume_vectors):
    similarities = similarity_matrix[i]
    top_indices = np.argsort(similarities)[-top_n:][::-1]
    top_jobs = [df1['Cleaned_Job_Description'].tolist()[idx] for idx in top_indices]
    recommendations.append(top_jobs)
6. Evaluate the results:
python
Copy
Edit
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

def evaluate_model(similarity_matrix, threshold=0.7):
    y_true = [1] * len(similarity_matrix)  # All samples are relevant
    y_pred = [1 if any(sim > threshold for sim in similarities) else 0 for similarities in similarity_matrix]
    
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    return precision, recall, f1, accuracy

precision, recall, f1, accuracy = evaluate_model(similarity_matrix)
7. Visualize the similarity distribution:
python
Copy
Edit
import seaborn as sns
import matplotlib.pyplot as plt

def visualize_similarity_distribution(similarity_scores, threshold=0.7):
    plt.figure(figsize=(10, 6))
    sns.histplot(similarity_scores, bins=30, kde=True, color='b')
    plt.axvline(x=threshold, color='r', linestyle='--', label=f'Threshold ({threshold})')
    plt.title("Distribution of Cosine Similarity Scores", fontsize=16)
    plt.xlabel("Cosine Similarity Score", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.legend()
    plt.show()

visualize_similarity_distribution(similarity_matrix.flatten())


Contributing
Contributions are always welcome! Feel free to open an issue or create a pull request for bug fixes, new features, or improvements.

License
This project is licensed under the MIT License. See the LICENSE file for details.
