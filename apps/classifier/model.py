import sqlite3

class ModelImage:
    def __init__(self, id, full_path):
        self._full_path = full_path
        self.id = id
        self.metadata = {}

class Model:
    def __init__(self, sqlite_database_path="metadata.db"):
        self._sqlite_init(sqlite_database_path)
    
    def _sqlite_init(self, database_path):
        """
        Initializes sqlite for usage
        """
        self.sqlite_connection = sqlite3.connect(database_path)
        self.sqlite_cursor = self.sqlite_connection.cursor()
        self._ensure_sqlite_schema()

    def _ensure_sqlite_schema(self):
        """
        Creates all key tables in the sqlite db if they do not exist
        """
        self.sqlite_cursor.execute('''
            CREATE TABLE IF NOT EXISTS image_link (
                id INTEGER PRIMARY KEY,
                filename TEXT,
                metadata TEXT
            )
                                   
            CREATE TABLE IF NOT EXISTS cathegory_directories (
                id INTEGER PRIMARY KEY,
                relative_path TEXT,
            )
        ''')
        self.sqlite_connection.commit()

    def _sqlite_close(self):
        self.sqlite_connection.close()

    def get_image_metadata(self, filepath):
        pass

    def cathegorize_image(self, image):
        pass

    def create_cathegory(self):
        

    def get_cathegory_list(self):
        pass

    def close(self):
        self._sqlite_close()
