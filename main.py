import streamlit as st
import joblib
import asyncio

from nlp import preprocess_input
from gpt_connector import GPTConnector


# Load the trained SVM model and the TF-IDF vectorizer
svm_model = joblib.load('svm/svm_model.pkl')
vectorizer = joblib.load('svm/tfidf_vectorizer.pkl')

# Streamlit UI
col1, col2 = st.columns([3, 1])
with col1:
    st.title("AI Tutor")
with col2:
    st.image("PBG mark2 color.svg", width=150)  # Increased size from 100 to 150

st.write("This app tokenize and lemmatize user input and classify questions into categories such as Factual,"
         " Conceptual, and Other. It then finally passes it to Groq to get the answer")

# Input for user to enter a question
question_input = st.text_input("Enter a question:")

def classify_question(question):
    # Step 1: Preprocess the question (tokenization and lemmatization)
    preprocessed_question = preprocess_input(question)

    # Step 2: Vectorize the preprocessed question using the loaded TF-IDF vectorizer
    vectorized_question = vectorizer.transform([preprocessed_question])  # Vectorize the input

    # Step 3: Predict the category using the loaded SVM model
    predicted_category = svm_model.predict(vectorized_question)[0]

    return predicted_category


async def get_answer_async(question: str):
    # Call the classify_question function with the user's input
    predicted_category = classify_question(question)

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
    return {"question": question, "predicted_category": predicted_category, "answer": answer}


# Synchronous function to call the async function
def get_answer(question: str):
    return asyncio.run(get_answer_async(question))


# When the user submits a question
if st.button("Submit"):
    if question_input:
        # Classify the question
        answer = get_answer(question_input)

        # Display the result
        st.write(f"**Answer:** {answer.get('answer')}")
    else:
        st.write("Please enter a question")
