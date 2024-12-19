import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer

# Path to the database
db_path = r"D:\EPR Data\Updated db'\knowledge_base.db"

# Load the Sentence-BERT model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Connect to the database
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Function to check if a column exists
def column_exists(cursor, table_name, column_name):
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [info[1] for info in cursor.fetchall()]  # Column names are in the second field
    return column_name in columns

# Step 1: Check and add new columns if not present
if not column_exists(cursor, "ValidatedQA", "question_embedding"):
    cursor.execute("ALTER TABLE ValidatedQA ADD COLUMN question_embedding BLOB")
if not column_exists(cursor, "ValidatedQA", "answer_embedding"):
    cursor.execute("ALTER TABLE ValidatedQA ADD COLUMN answer_embedding BLOB")
conn.commit()

# Step 2: Fetch questions and answers
cursor.execute("SELECT id, question, answer FROM ValidatedQA")
rows = cursor.fetchall()

# Step 3: Generate embeddings
for row in rows:
    question_id, question_text, answer_text = row
    try:
        # Generate embeddings
        question_embedding = model.encode(question_text)
        question_embedding = question_embedding / np.linalg.norm(question_embedding)  # Normalize
        answer_embedding = model.encode(answer_text)
        answer_embedding = answer_embedding / np.linalg.norm(answer_embedding)  # Normalize

        # Serialize embeddings to BLOB
        question_blob = question_embedding.tobytes()
        answer_blob = answer_embedding.tobytes()

        # Update the database
        cursor.execute("""
            UPDATE ValidatedQA 
            SET question_embedding = ?, answer_embedding = ?
            WHERE id = ?
        """, (question_blob, answer_blob, question_id))
    except Exception as e:
        print(f"Error processing ID {question_id}: {e}")

# Step 4: Commit changes and close the database
conn.commit()
conn.close()

print("Embeddings for questions and answers generated and updated successfully.")
