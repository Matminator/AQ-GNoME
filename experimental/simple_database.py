import sqlite3
import pickle
from pathlib import Path

class simple_database:
    def __init__(self, db_name, make_new_db = True):
        data_path = Path(__file__).parent / 'data' / db_name
        root_path = Path(__file__).parent / db_name
        if data_path.exists():
            db_path = data_path
        elif root_path.exists():
            db_path = root_path
        else:
            db_path = data_path  # default location for new DBs
        if not db_path.exists() and not make_new_db:
            raise FileNotFoundError(f"Database not found at {data_path} or {root_path}")

        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        self._create_table()

    def _create_table(self):
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS entries (
                id TEXT PRIMARY KEY,
                entry BLOB NOT NULL
            )
        ''')
        self.conn.commit()

    def get_num_entries(self):
        self.cursor.execute("SELECT COUNT(*) FROM entries;")
        count = self.cursor.fetchone()[0]
        return count

    def insert_entry(self, entry_id: str, entry):
        pickled_entry = pickle.dumps(entry, protocol=pickle.HIGHEST_PROTOCOL)
        self.cursor.execute(
            "INSERT OR REPLACE INTO entries (id, entry) VALUES (?, ?)",
            (entry_id, pickled_entry)
        )
        self.conn.commit()

    def get_entry(self, entry_id: str):
        self.cursor.execute("SELECT entry FROM entries WHERE id = ?", (entry_id,))
        row = self.cursor.fetchone()
        if row:
            return pickle.loads(row[0])
        return None

    def list_entry_ids(self) -> list[str]:
        self.cursor.execute("SELECT id FROM entries")
        return [row[0] for row in self.cursor.fetchall()]

    def delete_entry(self, entry_id: str, are_you_sure: bool = False):
        """Delete an entry from the database by its ID."""
        if not are_you_sure:
            raise ValueError("You must set 'are_you_sure' to True to delete an entry.")
        else:
            self.cursor.execute("DELETE FROM entries WHERE id = ?", (entry_id,))
            self.conn.commit()

    def close(self):
        self.conn.close()
