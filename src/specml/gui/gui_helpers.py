import os

def validate_file(file_path):
    """Helper function to check if the selected file exists and is valid."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found!")
    return True
