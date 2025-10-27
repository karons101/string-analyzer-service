import hashlib
from collections import Counter
import re
from typing import Dict, List, Optional
from datetime import datetime

# --- CRITICAL FIX: Suppress SQLAlchemy Deprecation Warnings ---
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

# --- FASTAPI & DEPENDENCY IMPORTS ---
from fastapi import FastAPI, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, Column, Integer, String, Boolean, JSON, select, DateTime, func, and_
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from pydantic import BaseModel, Field

# --- DATABASE CONFIGURATION ---
# Change from file-based to in-memory:
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:" 
# NOTE: Data will be lost on every deploy, but the service will start!
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)

# Create a SessionLocal class to establish database sessions
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for the SQL Models
Base = declarative_base()

# Dependency function to get a database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ----------------------------------------------------
# --- DATABASE MODEL (SQLAlchemy) ---
class StringAnalysis(Base):
    __tablename__ = "string_analysis"

    id = Column(Integer, primary_key=True, index=True)
    value = Column(String, index=True, nullable=False)
    sha256_hash = Column(String, unique=True, index=True, nullable=False) # Unique Constraint
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    length = Column(Integer)
    is_palindrome = Column(Boolean)
    unique_characters = Column(Integer)
    word_count = Column(Integer)
    character_frequency_map = Column(JSON)


# ----------------------------------------------------
# --- PYDANTIC SCHEMAS ---
class StringInput(BaseModel):
    value: str = Field(..., description="The string to be analyzed.", min_length=1)

class AnalysisProperties(BaseModel):
    length: int
    is_palindrome: bool
    unique_characters: int
    word_count: int
    sha256_hash: str
    character_frequency_map: Dict

    class Config:
        orm_mode = True

class StringAnalysisOut(BaseModel):
    id: str = Field(..., alias="sha256_hash", description="SHA-256 hash of the string.")
    value: str
    created_at: datetime
    properties: AnalysisProperties

    class Config:
        orm_mode = True
        allow_population_by_field_name = True

    @classmethod
    def build_response(cls, db_object: StringAnalysis) -> 'StringAnalysisOut':
        """Constructs the nested Pydantic object from a flat SQLAlchemy object."""
        properties_data = AnalysisProperties.from_orm(db_object)
        return cls(
            id=db_object.sha256_hash,
            value=db_object.value,
            created_at=db_object.created_at,
            properties=properties_data
        )
        
class StringListOut(BaseModel):
    count: int = Field(..., description="The total number of results returned.")
    filters: Dict[str, str] = Field(..., description="The filters applied to the query.") 
    data: List[StringAnalysisOut] = Field(..., description="The list of analysis results.")


# --- Initialize FastAPI App ---
app = FastAPI(title="String Analyzer Service")

# ----------------------------------------------------
# --- Core Analysis Logic (No Change) ---
def analyze_string(value: str) -> Dict:
    """Computes all required properties for a given string."""

    # 1. length
    length = len(value)

    # 2. is_palindrome (case-insensitive, non-alphanumeric ignored)
    normalized = re.sub(r'[^a-zA-Z0-9]', '', value).lower()
    is_palindrome = normalized == normalized[::-1]

    # 3. unique_characters
    unique_characters = len(set(value))

    # 4. word_count
    word_count = len(re.findall(r'\S+', value))

    # 5. sha256_hash (used as ID for deduplication)
    sha256_hash = hashlib.sha256(value.encode('utf-8')).hexdigest()

    # 6. character_frequency_map
    character_frequency_map = dict(Counter(value))

    return {
        "length": length,
        "is_palindrome": is_palindrome,
        "unique_characters": unique_characters,
        "word_count": word_count,
        "sha256_hash": sha256_hash,
        "character_frequency_map": character_frequency_map,
    }

# ----------------------------------------------------
# --- NLP HELPER FUNCTION (FINAL, FLEXIBLE REGEX) ---

def parse_natural_language_filters(query: str) -> Dict[str, str]:
    """
    Translates a simple natural language query into a dictionary of filters.
    
    Supports: palindrome, length, word count.
    """
    filters = {}
    query = query.lower()

    # 1. Palindrome check
    if "palindrome" in query:
        if "not palindrome" in query or "non-palindrome" in query:
            filters["is_palindrome"] = "false"
        else:
            filters["is_palindrome"] = "true"
            
    # Simple keyword match for 'palindromes'
    if "palindromes" in query and "is_palindrome" not in filters:
        filters["is_palindrome"] = "true"

    # 2. Length filters (using flexible regex)
    # The '.*?' allows optional words between 'length' and the operator/value
    length_match = re.search(r'length.*? (greater than|less than|of|equals) (\d+)', query)
    if length_match:
        operator = length_match.group(1)
        value = int(length_match.group(2))
        
        if operator == "greater than":
            filters["min_length"] = str(value + 1)
        elif operator == "less than":
            filters["max_length"] = str(value - 1)
        elif operator in ["of", "equals"]:
            filters["min_length"] = str(value)
            filters["max_length"] = str(value)
            
    # 3. Word count filters (FINAL, FLEXIBLE REGEX)
    # The '.*?' allows optional words between 'word count' and the operator/value (e.g., 'with a')
    word_match = re.search(r'word count.*? (greater than|less than|of|equals) (\d+)', query)
    if word_match:
        operator = word_match.group(1)
        value = int(word_match.group(2))
        
        if operator == "greater than":
            filters["min_word_count"] = str(value + 1) 
        elif operator == "less than":
            filters["max_word_count"] = str(value - 1)
        elif operator in ["of", "equals"]:
            filters["min_word_count"] = str(value)
            filters["max_word_count"] = str(value)

    return filters


def apply_filters_to_query(query, applied_filters):
    """Applies a dictionary of string filters (from Query params or NLP) to a SQLAlchemy statement."""
    
    # Cast boolean filter strings to actual booleans
    is_palindrome_filter = applied_filters.get("is_palindrome")
    if is_palindrome_filter is not None:
        bool_val = is_palindrome_filter.lower() == 'true'
        query = query.filter(StringAnalysis.is_palindrome == bool_val)

    # Cast integer filters
    min_length = applied_filters.get("min_length")
    if min_length is not None and min_length.isdigit():
        query = query.filter(StringAnalysis.length >= int(min_length))

    max_length = applied_filters.get("max_length")
    if max_length is not None and max_length.isdigit():
        query = query.filter(StringAnalysis.length <= int(max_length))

    min_word_count = applied_filters.get("min_word_count")
    if min_word_count is not None and min_word_count.isdigit():
        query = query.filter(StringAnalysis.word_count >= int(min_word_count))
        
    max_word_count = applied_filters.get("max_word_count")
    if max_word_count is not None and max_word_count.isdigit():
        query = query.filter(StringAnalysis.word_count <= int(max_word_count))
        
    return query

# ----------------------------------------------------
# --- API ENDPOINT: POST (Analyze and Deduplicate) ---
@app.post("/strings", response_model=StringAnalysisOut, status_code=status.HTTP_201_CREATED)
def analyze_and_save_string_api(
    string_input: StringInput,
    db: Session = Depends(get_db)
):
    """
    Analyzes a string, checking for duplicates by SHA256 hash.
    Saves a new record if the string has not been analyzed before.
    """
    # 1. Run the core analysis logic
    analysis_results = analyze_string(string_input.value)
    current_hash = analysis_results["sha256_hash"]

    # 2. Check the database for an existing record with this hash
    stmt = select(StringAnalysis).filter_by(sha256_hash=current_hash)
    db_analysis = db.scalar(stmt)

    if db_analysis:
        print(f"--- INFO: Hash {current_hash[:8]} found. Returning 409 Conflict.")
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="String already exists in the system"
        )
    
    # 4. If not found, create and save a new database record
    print(f"--- INFO: Hash {current_hash[:8]} NOT found. Creating new record.")
    db_analysis = StringAnalysis(
        value=string_input.value,
        **analysis_results
    )
    
    db.add(db_analysis)
    db.commit()
    db.refresh(db_analysis)

    return StringAnalysisOut.build_response(db_analysis)


# ----------------------------------------------------
# --- API ENDPOINT: GET (Retrieve All with Filtering) ---
@app.get("/strings", response_model=StringListOut)
def get_all_analyses(
    db: Session = Depends(get_db),
    # QUERY PARAMETERS for filtering
    is_palindrome: Optional[bool] = Query(None, description="Filter by palindrome status."),
    min_length: Optional[int] = Query(None, ge=1, description="Filter by minimum string length."),
    max_length: Optional[int] = Query(None, ge=1, description="Filter by maximum string length."),
    min_word_count: Optional[int] = Query(None, ge=1, description="Filter by minimum word count."),
):
    """
    Retrieves all stored string analysis records from the database, with optional query filtering.
    """
    # 1. Start building the SQLAlchemy query
    query = select(StringAnalysis).order_by(StringAnalysis.id.desc())
    
    # 2. Collect applied filters and convert them to strings for the response
    applied_filters = {}
    if is_palindrome is not None:
        applied_filters["is_palindrome"] = str(is_palindrome)
    if min_length is not None:
        applied_filters["min_length"] = str(min_length)
    if max_length is not None:
        applied_filters["max_length"] = str(max_length)
    if min_word_count is not None:
        applied_filters["min_word_count"] = str(min_word_count)

    # 3. Apply filters using the new helper function (since filters are already in string format)
    query = apply_filters_to_query(query, applied_filters)

    # 4. Execute the query
    results = db.scalars(query).all()
    
    # 5. Build and return the structured response
    data = [StringAnalysisOut.build_response(r) for r in results]
    
    return StringListOut(
        count=len(data),
        filters=applied_filters,
        data=data
    )


# ----------------------------------------------------
# --- API ENDPOINT: GET (Filter by Natural Language) ---
@app.get("/strings/filter-by-natural-language", response_model=StringListOut)
def filter_by_nlp(
    query: str = Query(..., description="Natural language query string (e.g., 'palindromes longer than 5')"),
    db: Session = Depends(get_db)
):
    """
    Retrieves string analysis records filtered by a natural language query.
    """
    # 1. Parse the natural language query into structured filters
    applied_filters = parse_natural_language_filters(query)

    if not applied_filters:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not parse any valid filters from the natural language query."
        )

    # 2. Start building the SQLAlchemy query
    db_query = select(StringAnalysis).order_by(StringAnalysis.id.desc())
    
    # 3. Apply filters using the helper function
    db_query = apply_filters_to_query(db_query, applied_filters)

    # 4. Execute the query
    results = db.scalars(db_query).all()
    
    # 5. Build and return the structured response
    data = [StringAnalysisOut.build_response(r) for r in results]
    
    return StringListOut(
        count=len(data),
        filters=applied_filters,
        data=data
    )


# ----------------------------------------------------
# --- API ENDPOINT: GET (Retrieve Specific String) ---
@app.get("/strings/{string_hash}", response_model=StringAnalysisOut)
def get_specific_analysis(string_hash: str, db: Session = Depends(get_db)):
    """
    Retrieves a specific stored string analysis record by its SHA256 hash.
    """
    # 1. Look up the record by the sha256_hash provided in the path
    stmt = select(StringAnalysis).filter_by(sha256_hash=string_hash)
    db_analysis = db.scalar(stmt)
    
    # 2. Handle 404 Not Found error
    if not db_analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="String not found in the system"
        )

    # 3. Return the ORM object using the custom builder method
    return StringAnalysisOut.build_response(db_analysis)


# ----------------------------------------------------
# --- API ENDPOINT: DELETE (Remove Specific String) ---
@app.delete("/strings/{string_hash}", status_code=status.HTTP_204_NO_CONTENT)
def delete_specific_analysis(string_hash: str, db: Session = Depends(get_db)):
    """
    Deletes a specific stored string analysis record by its SHA256 hash.
    Returns 204 No Content on success, or 404 Not Found if the string doesn't exist.
    """
    # 1. Look up the record by the sha256_hash provided in the path
    stmt = select(StringAnalysis).filter_by(sha256_hash=string_hash)
    db_analysis = db.scalar(stmt)
    
    # 2. Handle 404 Not Found error
    if not db_analysis:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="String not found in the system"
        )

    # 3. Delete the record and commit
    db.delete(db_analysis)
    db.commit()

    # 4. Return 204 No Content (handled by the status_code decorator)
    return


# --- Root Endpoint for Health Check ---
@app.get("/")
def read_root():
    """Simple root endpoint to confirm API is running."""
    return {"status": "ok", "service": "String Analyzer Service"}


# ----------------------------------------------------
# --- DATABASE INITIALIZATION ---
# Create database tables (in-memory SQLite table)
Base.metadata.create_all(bind=engine)

# ----------------------------------------------------
# --- EXTREME OVERRIDE: MANUAL RUN BLOCK ---
# This block forces the app to listen on port 8000, which Railway should pick up.
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)