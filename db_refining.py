import os
import spacy
import pdfplumber
import sqlite3
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
from collections import deque
from nltk.tokenize import sent_tokenize

# Load NLP models
nlp = spacy.load("en_core_web_md")
qa_model = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# Initialize conversation history for context retention
conversation_history = deque(maxlen=5)

# Database initialization
def init_db(db_path):
    if not os.path.exists(db_path):
        print(f"Creating database at: {db_path}")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ValidatedQA (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT UNIQUE,
            answer TEXT,
            embedding BLOB
        )
    """)
    conn.commit()
    return conn

# Save to database with embedding
def save_to_db(conn, question, answer, embedding):
    cursor = conn.cursor()
    cursor.execute("INSERT OR REPLACE INTO ValidatedQA (question, answer, embedding) VALUES (?, ?, ?)", 
                   (question, answer, embedding.tobytes()))
    conn.commit()

# Retrieve embedding and answer from database
def get_answer_from_db(conn, query_vector, threshold=0.7):
    cursor = conn.cursor()
    cursor.execute("SELECT question, answer, embedding FROM ValidatedQA")
    rows = cursor.fetchall()
    
    if not rows:
        return None, 0
    
    embeddings = [np.frombuffer(row[2], dtype=np.float32) for row in rows]
    answers = [row[1] for row in rows]
    
    scores = [cosine_similarity([query_vector], [emb])[0][0] for emb in embeddings]
    max_score_idx = np.argmax(scores)
    
    if scores[max_score_idx] >= threshold:
        return answers[max_score_idx], scores[max_score_idx]
    return None, max(scores)

# Extract and tokenize PDF content
def extract_and_tokenize_pdf(pdf_path):
    sections = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                sentences = sent_tokenize(text)
                sections.extend(sentences)
    return sections

# Summarize and vectorize sections
def summarize_sections(sections, max_length=300, min_length=100):
    summaries = []
    for section in sections:
        summary = section[:max_length].rsplit(' ', 1)[0] + "..." if len(section) > max_length else section
        if len(summary) < min_length:
            summary += " This section appears to be short; additional context may be required."
        summaries.append(summary)
    return summaries


def setup_vectorizer(summaries):
    vectorizer = TfidfVectorizer()
    summary_vectors = vectorizer.fit_transform(summaries)
    return vectorizer, summary_vectors

# Retrieve relevant summary from PDF
def find_relevant_summary(query, vectorizer, summary_vectors, summaries, num_sections=3):
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, summary_vectors).flatten()
    best_indices = np.argsort(scores)[-num_sections:]  # Fetch top 'num_sections' scores
    combined_summary = "\n".join([summaries[idx] for idx in reversed(best_indices)])
    return combined_summary


# Generate answer using QA model
def generate_ai_answer(context, query):
    extended_context = f"Context:\n{context}\n\nQuery:\n{query}"
    try:
        response = qa_model(question=query, context=extended_context)
        return response.get('answer', "AI failed to generate a valid answer.")
    except Exception as e:
        return f"AI encountered an error: {e}. Consider refining the query or context."

# Process text with spaCy
def process_text_with_spacy(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if token.is_alpha and not token.is_stop])

# Main model logic
def ask_question(conn, query, vectorizer, summary_vectors, summaries):
    query_processed = process_text_with_spacy(query)
    query_vector = nlp(query_processed).vector

    # Retrieve database answer and refine it
    db_answer, db_score = get_answer_from_db(conn, query_vector)
    if db_answer:
        db_answer = refine_database_answer(db_answer, query)

    # Retrieve larger context from PDF
    pdf_summary = find_relevant_summary(query_processed, vectorizer, summary_vectors, summaries)

    # Generate AI-based answer with enriched context
    ai_context = f"Database Answer:\n{db_answer}\n\nPDF Context:\n{pdf_summary}" if db_answer else pdf_summary
    ai_answer = generate_ai_answer(ai_context, query)

    # Merge Database and PDF answers into a cohesive Combined Answer
    combined_answer = merge_combined_answer(db_answer, pdf_summary)

    return db_answer, pdf_summary, ai_answer, combined_answer


def merge_combined_answer(db_answer, pdf_summary):
    """
    Combines database and PDF answers into a single cohesive response.
    Adds "According to CPCB" for PDF answers to enhance credibility.
    """
    combined = ""

    if db_answer:
        combined += f"The Plastic Waste Management Rules are described as follows:\n\n{db_answer.strip()}\n"

    if pdf_summary:
        combined += f"\nAdditionally, according to CPCB:\n\n{pdf_summary.strip()}"

    # Ensure combined answer has proper formatting
    return combined.strip()



def refine_database_answer(answer, query):
    if len(answer.split()) < 50:  # Ensure minimum word count
        additional_info = (
            "Plastic Waste Management Rules focus on addressing the environmental hazards of plastic waste. "
            "These rules mandate proper segregation, collection, and recycling of plastic waste, while assigning "
            "responsibilities to producers and brand owners under Extended Producer Responsibility (EPR)."
        )
        answer = f"{answer} {additional_info}"
    return answer



# Interaction loop
def start_interaction(conn, vectorizer, summary_vectors, summaries):
    print("Choose Mode: 1) Ask Questions, 2) Validate Database")
    print("Type 'exit' anytime to quit.")
    mode = input("Enter mode (1 or 2): ").strip()

    if mode.lower() == 'exit':
        print("Exiting...")
        return

    if mode not in ["1", "2"]:
        print("Invalid mode selected. Please choose 1 or 2.")
        return

    mode = int(mode)
    if mode == 1:
        print("Start asking your questions. Type 'exit' to quit.")
        while True:
            user_query = input("You: ").strip()
            if user_query.lower() == 'exit':
                print("Exiting...")
                break

            db_ans, pdf_ans, ai_ans, combined_ans = ask_question(conn, user_query, vectorizer, summary_vectors, summaries)

            print("\nAnswers:\n")
            print("A) Database:\n")
            print(f"{db_ans or 'No match'}\n")
            print("------------------------------------------------------\n")

            print("B) According to CPCB:\n")
            print(f"{pdf_ans or 'No match'}\n")
            print("------------------------------------------------------\n")

            print("C) AI:\n")
            print(f"{ai_ans or 'No match'}\n")
            print("------------------------------------------------------\n")

            if combined_ans:
                print("D) Combined Answer:\n")
                print(f"{combined_ans}\n")
                print("------------------------------------------------------\n")

            print("E) None of the above\n")

            choice = input("Choose the correct answer (A, B, C, D, E): ").strip().upper()
            if choice == 'EXIT':
                print("Exiting...")
                break

            if choice == 'E':
                correct_answer = input("Enter the correct answer: ").strip()
                save_to_db(conn, user_query, correct_answer, nlp(user_query).vector)
            elif choice in ['A', 'B', 'C', 'D']:
                chosen_ans = {'A': db_ans, 'B': pdf_ans, 'C': ai_ans, 'D': combined_ans}[choice]
                save_to_db(conn, user_query, chosen_ans, nlp(user_query).vector)
            else:
                print("Invalid choice! Please try again.")

    
    elif mode == 2:
        cursor = conn.cursor()
        cursor.execute("SELECT question FROM ValidatedQA")
        questions = cursor.fetchall()
        
        if not questions:
            print("No questions found in the database for validation.")
            return

        print("Enter the starting question number (or type 'exit' to quit):")
        start_index = input("Start from question number: ").strip()
        if start_index.lower() == 'exit':
            print("Exiting...")
            return

        if not start_index.isdigit() or int(start_index) < 1 or int(start_index) > len(questions):
            print("Invalid question number.")
            return
        
        start_index = int(start_index) - 1
        for i, (question,) in enumerate(questions[start_index:], start=start_index + 1):
            print("******************************************************\n")
            print(f"\nQuestion {i}: {question}")
            print("******************************************************\n")
            db_ans, pdf_ans, ai_ans, combined_ans = ask_question(conn, question, vectorizer, summary_vectors, summaries)

            print(f"A) Database: {db_ans or 'No match'}")
            print("------------------------------------------------------\n")
            print(f"B) PDF: {pdf_ans or 'No match'}")
            print("------------------------------------------------------\n")
            print(f"C) AI: {ai_ans or 'No match'}")
            print("------------------------------------------------------\n")
            if combined_ans:
                print(f"D) Combined Answer: {combined_ans}")
                print("------------------------------------------------------\n")
            print("E) None of the above")

            choice = input("Choose the correct answer (A, B, C, D, E): ").strip().upper()
            if choice == 'EXIT':
                print("Exiting...")
                break
            
            if choice == 'E':
                correct_answer = input("Enter the correct answer: ").strip()
                save_to_db(conn, question, correct_answer, nlp(question).vector)
            elif choice in ['A', 'B', 'C', 'D']:
                chosen_ans = {'A': db_ans, 'B': pdf_ans, 'C': ai_ans, 'D': combined_ans}[choice]
                save_to_db(conn, question, chosen_ans, nlp(question).vector)
            else:
                print("Invalid choice! Please try again.")


# Initialize
if __name__ == "__main__":
    try:
        # Define a static path for the database
        db_path = r"D:\EPR Data\Updated db'\knowledge_base.db"

        # Initialize the database
        conn = init_db(db_path)

        # Define the static path for the PDF
        pdf_path = r"D:\EPR Data\Test_EPR_Data.pdf"

        # Verify PDF path
        if not os.path.exists(pdf_path):
            print(f"Error: PDF file not found at {pdf_path}")
            exit()

        # Extract and tokenize PDF content
        pdf_sections = extract_and_tokenize_pdf(pdf_path)
        if not pdf_sections:
            print("Error: No content extracted from the PDF.")
            exit()

        # Summarize and vectorize the PDF content
        pdf_summaries = summarize_sections(pdf_sections)
        vectorizer, summary_vectors = setup_vectorizer(pdf_summaries)

        # Start interaction with the user
        start_interaction(conn, vectorizer, summary_vectors, pdf_summaries)

    except Exception as e:
        print(f"An error occurred: {e}")
