import pandas as pd
import json

def load_json_lines(file_path):
    """
    Reads a JSONL file and returns a list of dictionaries.
    """
    with open(file_path, 'r') as file:
        return [json.loads(line) for line in file]

def get_target_columns():
    """
    Returns the strict list of technical specifications to extract.
    """
    return [
        "Brand", "Model", "Reference number", "Movement", "Case material",
        "Bracelet material", "Year of production", "Scope of delivery",
        "Gender", "Location", "Price", "Availability", "Caliber/movement",
        "Base caliber", "Power reserve", "Number of jewels", "Case diameter",
        "Water resistance", "Bezel material", "Crystal", "Dial", 
        "Dial numerals", "Bracelet color", "Clasp", "Clasp material"
    ]

def clean_to_single_line(value, max_length=100):
    """
    Truncates text to the first line and enforces a character limit.
    """
    if not value:
        return None
    
    # Keep only the first line of text
    first_line = str(value).split('\n')[0].strip()
    
    if len(first_line) > max_length:
        return first_line[:max_length-3] + "..."
        
    return first_line

def parse_line_to_pair(line):
    """
    Splits a line into a key and value, cleaning surrounding whitespace.
    """
    if ":" not in line:
        return None, None
        
    key, value = line.split(":", 1)
    return key.strip(), value.strip()

def extract_selected_specs(record, target_keys):
    """
    Filters the raw row list for specific key-value pairs and cleans them.
    """
    raw_lines = record.get("specifications") or record.get("data") or []
    specifications = {key: None for key in target_keys}
    
    for line in raw_lines:
        key, value = parse_line_to_pair(line)
        
        if key in target_keys:
            specifications[key] = clean_to_single_line(value)
            
    return specifications

def convert_jsonl_to_clean_csv(input_path, output_path):
    """
    Orchestrates the conversion of JSONL to a structured, one-line-per-cell CSV.
    """
    target_columns = get_target_columns()
    raw_data = load_json_lines(input_path)
    
    refined_records = []
    for record in raw_data:
        specs = extract_selected_specs(record, target_columns)
        specs["url"] = record.get("url")
        refined_records.append(specs)
    
    df = pd.DataFrame(refined_records, columns=target_columns + ["url"])
    df = df.drop_duplicates(subset="url")
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    convert_jsonl_to_clean_csv("scraped_watches.jsonl", "output.csv")