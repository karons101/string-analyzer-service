This is a RESTful API service built with FastAPI that analyzes strings, 
computes key linguistic and technical properties, and stores the results in a persistent SQLite database. 

It fully implements features like duplicate checking (via SHA-256 hash), flexible filtering using standard query parameters, 
and advanced filtering using Natural Language Processing (NLP).

This service was developed as part of the Backend Wizards — Stage 1 Task.

 Features:  
The service is built around a single StringAnalysis resource and provides the following capabilities:

String Analysis: Automatically calculates six properties for any submitted string: 
length, is_palindrome, unique_characters, word_count, sha256_hash, and character_frequency_map.Deduplication: 
Uses the SHA-256 hash of the string as a unique identifier. 
Attempting to submit a string that already exists results in a 409 Conflict error.
Query Filtering (GET /strings): Supports filtering by string properties using standard query parameters 
(e.g., is_palindrome, min_length, max_length, min_word_count).Natural Language Processing 
(NLP) Filtering (GET /strings/filter-by-natural-language): 
Translates conversational English queries into structured API filters 
(e.g., "palindromes with a word count greater than one").
Persistence: Uses SQLAlchemy with a SQLite database file (analyzer.db) 
for persistent storage.

 Setup and Local RunPrerequisitesYou need Python 3.8+ installed on your system.
1. Clone the Repository
git clone https://github.com/karons101/string-analyzer-service.git
cd string-analyzer-service

2. Install Dependencies I used a virtual environment, Bash# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use: .\venv\Scripts\activate

# Installed the required packages
# Ensure I had the requirements.txt, or install them manually:
pip install fastapi uvicorn 'pydantic[email]' sqlalchemy

3. I ran the ServiceThe application uses Uvicorn, an ASGI server, with the --reload flag for development.Bashuvicorn main:app --reload
The service will start and be accessible at: http://127.0.0.1:8000The interactive API documentation 
(Swagger UI) is available at: http://127.0.0.1:8000/docs🧭 API Endpoints

The service uses the SHA-256 hash of the string as its primary identifier for retrieval and deletion.
MethodEndpointDescriptionStatus CodesPOST/stringsAnalyzes a string and saves its properties. 
Checks for duplicates.201, 409GET/stringsRetrieves all analyses with optional query parameter filtering.200GET/strings/{string_hash}Retrieves a specific analysis by its SHA-256 hash.200, 404GET/strings/filter-by-natural-languageRetrieves analyses using a natural language query string.200, 400DELETE/strings/{string_hash}Deletes a specific analysis by its SHA-256 hash.204, 404:label: Natural Language Query ExamplesThe NLP endpoint supports flexible queries using keywords like "palindrome," "word count," and "length."Natural Language QueryInterpreted Filterspalindromes word count greater than 1is_palindrome=true, min_word_count=2strings longer than 10 charactersmin_length=11non-palindromes with length of 5is_palindrome=false, min_length=5, max_length=5(Note: The most reliable query format uses clear structure, e.g., palindromes word count greater than 1)📝 DependenciesThe main dependencies are: fastapi, uvicorn, SQLAlchemy, and pydantic.





