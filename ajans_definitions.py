from pymongo import MongoClient
from openai import OpenAI
from langchain.schema import Document


class AtlasClient:
    """
    Atlas (Mongo) class to run vector search algorithm.

    """

    def __init__(self, atlas_uri: str, dbname: str):
        self.mongodb_client = MongoClient(atlas_uri)
        self.database = self.mongodb_client[dbname]

    def ping(self) -> None:
        """Test connectivity to the Atlas instance."""
        self.mongodb_client.admin.command('ping')

    def get_collection(self, collection_name: str):
        return self.database[collection_name]

    def find(self, collection_name: str, filter: dict = {}, limit: int = 10) -> list:
        collection = self.database[collection_name]
        return list(collection.find(filter=filter, limit=limit))

    def vector_search(
        self,
        collection_name: str,
        index_name: str,
        attr_name: str,
        embedding_vector: list[float],
        limit: int = 5
    ) -> list:
        collection = self.database[collection_name]
        results = collection.aggregate([
            {
                '$vectorSearch': {
                    "index": index_name,
                    "path": attr_name,
                    "queryVector": embedding_vector,
                    "numCandidates": 50,
                    "limit": limit,
                }
            },
            {
                "$project": {
                    '_id': 1,
                    'Element_Description': 1,
                    'search_score': {"$meta": "vectorSearchScore"}
                }
            }
        ])
        return list(results)

    def close_connection(self) -> None:
        """Close the MongoDB connection."""
        self.mongodb_client.close()


class OpenAIClient:
    """
    OpenAI wrapper class for embeddings and completion.
    """

    def __init__(self, api_key: str) -> None:
        self.client = OpenAI(api_key=api_key)

    def get_embedding(self, text: str, model: str = "text-embedding-3-large") -> list[float]:
        """Generate an embedding for the given text."""
        cleaned = text.replace("\n", " ")
        resp = self.client.embeddings.create(
            input=[cleaned],
            model=model
        )
        return resp.data[0].embedding


# --- Document conversion functions with metadata ---

def job_to_document(row: dict) -> Document:
    """
    Convert a job record (from admin panel) to a langchain Document with metadata.
    """
    # Handle date fields that may be datetime objects
    deadline = row.get("deadline")
    deadline_str = (
        deadline.isoformat() if hasattr(deadline, "isoformat") else str(deadline)
    ) if deadline is not None else None

    return Document(
        page_content=row.get("description", ""),
        metadata={
            "job_id": row.get("job_id"),
            "description": row.get("description"),
            "budget": row.get("budget"),
            "deadline": deadline_str,
            "status": row.get("status"),
            "assigned_freelancer_id": row.get("assigned_freelancer_id"),
            "skill": row.get("skill"),
            "experience": row.get("experience"),
            "is_available": row.get("is_available"),
            "username": row.get("username"),
            "role": row.get("role"),
        }
    )


def freelancer_to_document(row: dict) -> Document:
    """
    Convert a freelancer record to a langchain Document with metadata.
    """
    name = row.get("name", "")
    skill = row.get("skill", "")
    return Document(
        page_content=f"{name} — {skill}",
        metadata={
            "freelancer_id": row.get("freelancer_id"),
            "name": name,
            "skill": skill,
            "experience": row.get("experience"),
            "is_available": row.get("is_available"),
        }
    )


def user_to_document(row: dict) -> Document:
    """
    Convert a user record to a langchain Document with metadata.
    """
    return Document(
        page_content=row.get("username", ""),
        metadata={
            "user_id": row.get("user_id"),
            "username": row.get("username"),
            "role": row.get("role"),
        }
    )
