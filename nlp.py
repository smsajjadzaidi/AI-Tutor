import spacy
# import nltk
# nltk.download('punkt_tab')


# Load SpaCy model for tokenization and lemmatization
nlp = spacy.load("en_core_web_sm")


def preprocess_input(text: str) -> str:
    # Tokenize and lemmatize the input question
    doc = nlp(text)
    lemmatized_tokens = [token.lemma_ for token in doc]
    return " ".join(lemmatized_tokens)


# Test the preprocessing function
if __name__ == "__main__":
    user_input = "What are the running benefits for runners who have been training regularly?"
    print(f"Original Input: {user_input}")
    processed_input = preprocess_input(user_input)
    print(f"Preprocessed Input: {processed_input}")
