import joblib
from fastapi import FastAPI
from pydantic import BaseModel

from nlp import preprocess_input
from gpt_connector import GPTConnector


# Load the trained SVM model and the TF-IDF vectorizer
svm_model = joblib.load('svm/svm_model.pkl')
vectorizer = joblib.load('svm/tfidf_vectorizer.pkl')


# Initialize FastAPI
app = FastAPI()


# Pydantic model for the input data
class Question(BaseModel):
    question: str


def classify_question(question):
    # Step 1: Preprocess the question (tokenization and lemmatization)
    preprocessed_question = preprocess_input(question)

    # Step 2: Vectorize the preprocessed question using the loaded TF-IDF vectorizer
    vectorized_question = vectorizer.transform([preprocessed_question])  # Vectorize the input

    # Step 3: Predict the category using the loaded SVM model
    predicted_category = svm_model.predict(vectorized_question)[0]

    return predicted_category


@app.post("/answer/")
async def get_answer(question: Question):
    # Call the classify_question function with the user's input
    predicted_category = classify_question(question.question)

    # Step 2: Adjust GPT-4 prompt based on the category
    match predicted_category:
        case "factual":
            prompt = f"Give a short and direct answer to: {question}"
        case "conceptual":
            prompt = f"Explain the concept behind: {question}"
        case "other":
            prompt = f"Analyze the tone of the question and answer accordingly: {question}"
        case _:
            return {"error": "category not recognised"}

    # Step 3: call GPT-4 to get the answer
    gpt_connector = GPTConnector(temperature=0.5)
    prompt += "\n Note: Do not add any insights about the question or its tone in the answer"
    answer = ""
    async for res in gpt_connector.get_gpt_response_stream(question=prompt):
        if res.answer:
            answer += res.answer

    if not answer:
        return {"error": "Something went wrong!"}

    # Return the predicted category as a JSON response
    return {"question": question.question, "predicted_category": predicted_category, "answer": answer}


@app.get("/")
def read_root():
    return {"message": "Welcome to the AI Tutor API"}


