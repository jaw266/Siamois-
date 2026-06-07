import sqlite3
from pathlib import Path

DB_PATH = Path("../outputs/results.db")

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

cursor.execute("""
    SELECT
        id,
        id_test,
        timestamp,
        prediction,
        classification_probability,
        change_area_ratio,
        severity,
        mask_path,
        overlay_path,
        received_at
    FROM detection_results
    ORDER BY id DESC
    LIMIT 20
""")

rows = cursor.fetchall()
conn.close()

print("\nDerniers résultats enregistrés:\n")

if not rows:
    print("Aucun résultat trouvé.")
else:
    for row in rows:
        print("--------------------------------")
        print("ID:", row[0])
        print("Test:", row[1])
        print("Timestamp:", row[2])
        print("Prediction:", row[3])
        print("Probability:", row[4])
        print("Change area ratio:", row[5])
        print("Severity:", row[6])
        print("Mask:", row[7])
        print("Overlay:", row[8])
        print("Received at:", row[9])