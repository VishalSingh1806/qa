import os
import asyncpg
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from flask import Flask, render_template, request, jsonify
from asgiref.wsgi import WsgiToAsgi
from flask_cors import CORS
import asyncio
import logging

# Flask app
app = Flask(__name__)
CORS(app)

# Load Sentence-BERT model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Database constants
DB_CONFIG = {
    'user': 'singh',
    'password': 'Tech123',
    'database': 'knowledge_base',
    'host': 'localhost',  # Replace with your server IP or hostname if needed
    'port': 5432
}

SIMILARITY_THRESHOLD = 0.7
logging.basicConfig(level=logging.DEBUG)

# Global variable for preloaded data
preloaded_data = []


# --- Database Repository ---
class DatabaseRepository:
    def __init__(self, db_config):
        self.db_config = db_config

    async def execute_query(self, query, params=None, fetch_one=False, fetch_all=False):
        """Execute a query and optionally fetch results."""
        conn = await asyncpg.connect(**self.db_config)
        try:
            if fetch_one:
                result = await conn.fetchrow(query, *params or [])
            elif fetch_all:
                result = await conn.fetch(query, *params or [])
            else:
                result = await conn.execute(query, *params or [])
            return result
        except asyncpg.PostgresError as e:
            logging.error(f"Database error: {e}")
            raise
        finally:
            await conn.close()


# Initialize repository
db_repo = DatabaseRepository(DB_CONFIG)


# --- Utility Functions ---
def compute_embedding(text):
    """Compute embedding using Sentence-BERT."""
    return model.encode(text).reshape(1, -1)  # Ensure 384 dimensions


async def preload_database():
    """Load all data from the database into memory."""
    global preloaded_data
    preloaded_data = await db_repo.execute_query(
        "SELECT question, answer, embedding FROM ValidatedQA", fetch_all=True
    )
    logging.debug("Database successfully preloaded.")
    logging.debug(f"Current preloaded data: {preloaded_data}")


async def initialize_database():
    """Initialize the database using schema and seed data files."""
    try:
        # Load and execute schema
        with open("schema.sql", "r") as schema_file:
            schema = schema_file.read()
        await db_repo.execute_query(schema)

        # Load and execute seed data
        with open("seed_data.sql", "r") as seed_file:
            seed_data = seed_file.read()
        await db_repo.execute_query(seed_data)

        logging.info("Database initialized successfully from SQL files.")
    except Exception as e:
        logging.error(f"Error initializing database from SQL files: {e}")
        raise


async def save_or_update_question(db_repo, question, answer, embedding):
    """Save or update a QA pair in the database."""
    try:
        # Check if the question exists in the database
        existing_entry = await db_repo.execute_query(
            "SELECT id FROM ValidatedQA WHERE question = $1",
            (question,),
            fetch_one=True,
        )

        if existing_entry:
            logging.debug(f"Updating existing question: {question}")
            await db_repo.execute_query(
                "UPDATE ValidatedQA SET answer = $1, embedding = $2 WHERE id = $3",
                (answer, embedding.tobytes(), existing_entry['id']),
            )
        else:
            logging.debug(f"Inserting new question: {question}")
            await db_repo.execute_query(
                "INSERT INTO ValidatedQA (question, answer, embedding) VALUES ($1, $2, $3)",
                (question, answer, embedding.tobytes()),
            )

        logging.debug("Successfully saved or updated question in database.")
        await preload_database()  # Reload data after every update
    except Exception as e:
        logging.error(f"Error saving or updating question in database: {e}")
        raise


async def fetch_best_match(user_embedding):
    """Query the preloaded data for the best match."""
    max_similarity = 0.0
    best_answer = None

    for row in preloaded_data:
        db_embedding_array = np.frombuffer(row['embedding'], dtype=np.float32).reshape(1, -1)
        similarity = cosine_similarity(user_embedding, db_embedding_array)[0][0]
        if similarity > max_similarity:
            max_similarity = similarity
            best_answer = row['answer']

    if max_similarity >= SIMILARITY_THRESHOLD:
        return best_answer, float(max_similarity)
    return None, 0.0


# --- Flask Routes ---
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/questions", methods=["GET"])
async def get_questions():
    """Fetch all questions from the preloaded data."""
    try:
        if not preloaded_data:
            await preload_database()
        questions = [{"question": row['question'], "answer": row['answer']} for row in preloaded_data]
        logging.debug(f"Fetched questions: {questions}")
        return jsonify({"questions": questions})
    except Exception as e:
        logging.error(f"Error fetching questions: {e}")
        return jsonify({"error": "Failed to fetch questions"}), 500


@app.route("/ask", methods=["POST"])
async def ask():
    """Handle a question and return the answer from the preloaded data."""
    try:
        data = request.json
        question = data.get("question", "").strip()
        if not question:
            return jsonify({"error": "Question cannot be empty"}), 400

        # Generate embedding for user query
        user_embedding = compute_embedding(question)

        # Query preloaded data for the best match
        db_answer, confidence = await fetch_best_match(user_embedding)

        # Return the database answer for validation
        return jsonify({
            "database": db_answer or "No match found in database.",
            "confidence": confidence
        })
    except Exception as e:
        logging.error(f"Error in /ask route: {e}")
        return jsonify({"error": "Internal server error"}), 500


@app.route("/add", methods=["POST"])
async def add_to_database():
    """Add or update a question-answer pair in the database."""
    try:
        data = request.json
        question = data.get("question", "").strip()
        answer = data.get("answer", "").strip()

        if not question or not answer:
            return jsonify({"error": "Both question and answer are required."}), 400

        # Compute embedding for the question
        embedding = compute_embedding(question)

        # Add or update the question-answer pair in the database
        await save_or_update_question(db_repo, question, answer, embedding)

        # Verify the update
        updated_entry = await db_repo.execute_query(
            "SELECT id, question, answer FROM ValidatedQA WHERE question = $1",
            (question,),
            fetch_one=True,
        )

        if not updated_entry:
            logging.error(f"Failed to update or find the question: {question}")
            return jsonify({"error": "Failed to update the question in the database."}), 500

        logging.debug(f"Updated entry: {updated_entry}")
        return jsonify({"message": "Answer submitted and saved successfully!", "updated_entry": dict(updated_entry)})
    except Exception as e:
        logging.error(f"Error in /add route: {e}")
        return jsonify({"error": "Internal server error"}), 500


# Wrap Flask app as ASGI for Uvicorn
asgi_app = WsgiToAsgi(app)

# --- Run App ---
if __name__ == "__main__":
    # Ensure the database table exists
    async def ensure_table():
        try:
            await initialize_database()
            await preload_database()  # Preload data on startup
        except Exception as e:
            logging.error(f"Error ensuring database table exists: {e}")

    asyncio.run(ensure_table())
    app.run(debug=True)
