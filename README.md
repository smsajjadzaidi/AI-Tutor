# AI-Tutor

**AI Tutor Bot** is an interactive web application designed to classify user questions into categories such as **factual**, **conceptual**, or **other**. Once a question is classified, the app generates an appropriate response using *Groq API**. This solution is built on **Streamlit** for easy UI interaction but also integrates **FastAPI** for backend flexibility in future expansions.

You can try out the app [here](https://groq-aitutor.streamlit.app/).

## Features

1. **Question Classification**:
    - Uses an **SVM (Support Vector Machine)** model to classify questions into predefined categories:
      - **Factual**: For questions requiring a direct and specific answer.
      - **Conceptual**: For questions requiring an explanation of a concept.
      - **Other**: For questions that don't fit into the other categories (e.g., open-ended, opinion-based).
      
2. **Groq Integration**:
    - After classification, the app generates responses using Groq with the llama-3.2-11b-vision-preview model based on the type of question.
    - Factual questions receive short, direct answers.
    - Conceptual questions receive detailed explanations.
    - Other questions are answered by analyzing the tone and intent behind the question.

3. **Preprocessing Pipeline**:
    - User input is preprocessed using **tokenization** and **lemmatization** before being fed into the classification model. This ensures the model accurately interprets the core meaning of the question.

4. **Streamlit UI**:
    - Provides an intuitive and easy-to-use interface where users can input questions and get real-time classified answers.
    - Currently deployed and hosted on **Streamlit** at [groq-aitutor.streamlit.app](https://groq-aitutor.streamlit.app/).

5. **FastAPI Integration**:
    - Although the current version runs on Streamlit, there’s also a **FastAPI** backend built-in, allowing for future flexibility and potential integration with more complex services.

---

## How the SVM Model was Trained

### Data Preparation:
We prepared a dataset consisting of **factual**, **conceptual**, and **other** questions. Each question was labeled according to its type:
- **Factual**: Objective questions that require specific information.
- **Conceptual**: Questions that require a deeper explanation of a concept.
- **Other**: Questions that do not fit into the factual or conceptual categories, including opinion-based or open-ended questions.

### Preprocessing:
The dataset was preprocessed using **NLP techniques** such as **tokenization** and **lemmatization** to clean the text before feeding it into the model. This preprocessing step ensures that only meaningful words are passed to the model, improving classification accuracy.

Steps involved:
1. **Tokenization**: Breaking down the text into individual words (tokens).
2. **Lemmatization**: Reducing words to their base form (e.g., "running" becomes "run") to ensure consistency.

### Training the SVM:
We trained the **Support Vector Machine (SVM)** classifier using the preprocessed dataset:
- **Vectorization**: We used **TF-IDF (Term Frequency-Inverse Document Frequency)** to convert the text into numerical vectors.
- **Training**: The SVM model was trained on labeled data to differentiate between factual, conceptual, and other questions. The model was tuned to handle questions of varying complexities.
- **Evaluation**: The model was evaluated on a test set, and fine-tuned to optimize its performance across different question types.

### Model Files:
- **SVM Model**: The trained SVM model is saved as `svm_model.pkl`.
- **TF-IDF Vectorizer**: The vectorizer used to convert text into vectors is saved as `tfidf_vectorizer.pkl`.

---

## How the Application Works

1. **User Input**: Users enter a question in the Streamlit app's input field.
2. **Classification**: The SVM model classifies the question into one of three categories: factual, conceptual, or other.
3. **Groq Response**: Based on the question type, a prompt is created and passed to Groq API, which generates the appropriate response.
4. **Answer Display**: The app displays the classified category and the Groq API response on the screen.

---

## Running Locally

### Prerequisites
- Python 3.7 or higher
- Install required packages via `requirements.txt`

```bash
pip install -r requirements.txt
```

Running the Streamlit App
To run the Streamlit app locally, use the following command:

```bash
streamlit run streamlit_app.py
```

This will launch the app in your default browser.

Running the FastAPI Backend
If you'd like to experiment with the FastAPI backend, use the following command to start the server:

```bash
uvicorn fastapi_app:app --reload
```
