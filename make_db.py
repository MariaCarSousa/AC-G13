# Puts all the CSV files in one SQLite database (outputs/hockey.db).
# Run from the project folder: python make_db.py

import sqlite3

from src.data import DATA_DIR, load_table

DB_PATH = "outputs/hockey.db"

conn = sqlite3.connect(DB_PATH)

for csv_file in sorted(DATA_DIR.glob("*.csv")):
    name = csv_file.stem
    df = load_table(name)
    df.to_sql(name, conn, if_exists="replace", index=False)
    print(f"{name}: {len(df)} rows")

# indexes to make the joins faster
conn.execute("CREATE INDEX IF NOT EXISTS idx_teams ON Teams (year, tmID)")
conn.execute("CREATE INDEX IF NOT EXISTS idx_coaches ON Coaches (coachID, year)")
conn.execute("CREATE INDEX IF NOT EXISTS idx_scoring ON Scoring (year, tmID)")
conn.execute("CREATE INDEX IF NOT EXISTS idx_goalies ON Goalies (year, tmID)")
conn.execute("CREATE INDEX IF NOT EXISTS idx_scoring_player ON Scoring (playerID, year)")
conn.execute("CREATE INDEX IF NOT EXISTS idx_awards_player ON AwardsPlayers (playerID, year)")

conn.commit()
conn.close()
print("Saved", DB_PATH)
