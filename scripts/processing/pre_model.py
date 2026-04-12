import pandas as pd

def filter_and_save_csv_columns(input_file_path, output_file_path, selected_columns):
    """
    Loads a CSV file, filters for specific columns, and saves the result.
    """
    source_dataframe = pd.read_csv(input_file_path)
    filtered_dataframe = source_dataframe[selected_columns]
    filtered_dataframe.to_csv(output_file_path, index=False)

if __name__ == "__main__":
    filter_and_save_csv_columns(
        input_file_path="data/csv/watches_classified.csv", 
        output_file_path="data/csv/filtered_for_inference_model.csv", 
        selected_columns=["Unnamed: 0", "Brand", "Model", "Case material", "Price", "Bezel material", "Dial"]
    )