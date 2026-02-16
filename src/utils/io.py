import csv
from pathlib import Path
from typing import Dict, List


def save_rows_to_csv(out_path: str, rows: List[Dict]) -> None:
    """
    This function helps to save a list of dict rows to a CSV file.
    It also creates parent directory if needed.
    """
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if len(rows) == 0:
        # still create an empty file with no header
        path.write_text("")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
