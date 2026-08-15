import dotenv
import os
from langchain_community.utilities import SQLDatabase

dotenv.load_dotenv()

DB_URI= os.getenv("DB_URI")

db = SQLDatabase.from_uri(DB_URI)

