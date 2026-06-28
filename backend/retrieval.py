import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Get directory path to playbooks relative to retrieval.py location
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLAYBOOKS_DIR = os.path.join(BASE_DIR, "data", "playbooks")

def match_playbook(query: str) -> dict:
    """
    Finds the best matching playbook in backend/data/playbooks/ based on cosine similarity of TF-IDF vectors.
    """
    if not os.path.exists(PLAYBOOKS_DIR):
        return {
            "playbook": "None",
            "score": 0.0,
            "text": f"Playbooks directory not found at: {PLAYBOOKS_DIR}"
        }

    playbooks = []
    filenames = []

    for filename in os.listdir(PLAYBOOKS_DIR):
        if filename.endswith(".txt"):
            filepath = os.path.join(PLAYBOOKS_DIR, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read().strip()
                    if content:
                        playbooks.append(content)
                        filenames.append(filename)
            except Exception as e:
                print(f"Error reading playbook {filename}: {e}")

    if not playbooks:
        return {
            "playbook": "None",
            "score": 0.0,
            "text": "No playbooks found in the playbooks directory."
        }

    # Vectorize docs + query
    documents = playbooks + [query]
    
    vectorizer = TfidfVectorizer(stop_words='english')
    try:
        tfidf_matrix = vectorizer.fit_transform(documents)
    except ValueError:
        # Happens if query and docs contain only stop words or are completely empty
        return {
            "playbook": filenames[0],
            "score": 0.0,
            "text": playbooks[0]
        }

    query_vector = tfidf_matrix[-1]
    playbook_vectors = tfidf_matrix[:-1]

    # Calculate cosine similarity
    similarities = cosine_similarity(query_vector, playbook_vectors)[0]
    
    # Get best match index
    best_idx = int(similarities.argmax())
    best_score = float(similarities[best_idx])

    return {
        "playbook": filenames[best_idx],
        "score": best_score,
        "text": playbooks[best_idx]
    }
