import csv
import re

def parse_tsf(tsf_path):
    series = {}
    with open(tsf_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line:
                continue
            # Ex: difficulty:2009-01-03 00-00-00:1,?,?,?,?,?,1,1,1,...
            match = re.match(r"(\w+):([0-9\- :]+):(.*)", line)
            if match:
                name, start_date, values = match.groups()
                values = values.split(',')
                series[name] = (start_date, values)
    return series

def write_csv(series, csv_path):
    # Find max length for padding
    max_len = max(len(vals[1]) for vals in series.values())
    # Prepare header
    header = ['date'] + list(series.keys())
    # Prepare rows
    rows = []
    # Get start dates and values for each series
    starts = {k: v[0] for k, v in series.items()}
    values = {k: v[1] for k, v in series.items()}
    # Build rows
    for i in range(max_len):
        row = []
        # Use the first series' start date as base (could be improved)
        base = list(series.keys())[0]
        date = starts[base]
        row.append(date if i == 0 else "")
        for k in series.keys():
            val_list = values[k]
            row.append(val_list[i] if i < len(val_list) else "")
        rows.append(row)
    # Write CSV
    with open(csv_path, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

if __name__ == "__main__":
    tsf_file = r"C:\Users\Utilisateur\Documents\Cour\Data Science\GROUPE-EXAM-FINAL\Predict_Bitcoin\Data\bitcoin_dataset_with_missing_values.tsf"
    csv_file = r"C:\Users\Utilisateur\Documents\Cour\Data Science\GROUPE-EXAM-FINAL\Predict_Bitcoin\Data\bitcoin_dataset.csv"
    series = parse_tsf(tsf_file)
    write_csv(series, csv_file)
    print(f"Conversion terminée : {csv_file}")