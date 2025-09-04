import os
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

def get_db_connection():
    return psycopg2.connect(DATABASE_URL)

# ---------- Application Logs ----------
def create_application_logs():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS application_logs (
            id SERIAL PRIMARY KEY,
            session_id TEXT NOT NULL,
            user_query TEXT NOT NULL,
            ai_response TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    cur.close()
    conn.close()

def insert_application_logs(session_id, user_query, ai_response):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('''
        INSERT INTO application_logs (session_id, user_query, ai_response)
        VALUES (%s, %s, %s)
    ''', (session_id, user_query, ai_response))
    conn.commit()
    cur.close()
    conn.close()

def get_chat_history(session_id):
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cur.execute('''
        SELECT user_query, ai_response
        FROM application_logs
        WHERE session_id = %s
        ORDER BY created_at
    ''', (session_id,))
    messages = []
    for row in cur.fetchall():
        messages.extend([
            {"role": "human", "content": row["user_query"]},
            {"role": "ai", "content": row["ai_response"]}
        ])
    cur.close()
    conn.close()
    return messages

# ---------- Document Store ----------
def create_document_store():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS document_store (
            id SERIAL PRIMARY KEY,
            filename TEXT NOT NULL,
            file_path TEXT,
            upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    cur.close()
    conn.close()

def insert_document_record(filename, file_path=None):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('''
        INSERT INTO document_store (filename, file_path)
        VALUES (%s, %s) RETURNING id
    ''', (filename, file_path))
    file_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()
    return file_id

def delete_document_record(file_id):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('DELETE FROM document_store WHERE id = %s', (file_id,))
    conn.commit()
    cur.close()
    conn.close()
    return True

def get_all_documents():
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)
    cur.execute('''
        SELECT id, filename, file_path, upload_timestamp
        FROM document_store
        ORDER BY upload_timestamp DESC
    ''')
    documents = cur.fetchall()
    cur.close()
    conn.close()
    return [dict(doc) for doc in documents]

# ---------- Init ----------
if __name__ == "__main__":
    create_application_logs()
    create_document_store()
    print("PostgreSQL tables created successfully.")