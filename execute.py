import csv
import multiprocessing
import sys

def test_code(path: str):
    run_codegen_test(path)

def run_codegen_test(file_path: str):
    """
    Reads a CSV file and prints the combined content of "Code" and "Test" columns.

    This function opens a CSV file, iterates through each row, and for each row,
    it concatenates the values from the 'Code' and 'Test' columns with two
    newline characters in between. The resulting string is then executed and prints the final score.

    Args:
        file_path (str): The full path to the input CSV file.

    Raises:
        FileNotFoundError: If the CSV file cannot be found at the specified path.
        KeyError: If the required columns "Code" or "Test" are not in the CSV file.
    """
    try:
        with open(file_path, mode='r', newline='', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            
            if 'Code' not in reader.fieldnames or 'Test' not in reader.fieldnames:
                raise KeyError("CSV must contain 'Code' and 'Test' columns.")

            print(f"--- Reading data from: {file_path} ---\n")
            
            i = 0
            score = 0
            for i, row in enumerate(reader):
                code_content = row['Code']
                test_content = row['Test']
                
                combined_output = f"{code_content}\n\n{test_content}"
                
                score += execute_code(combined_output)
            
            print(f"Final Score: {score}/{i+1}")

    except FileNotFoundError:
        print(f"Error: The file at '{file_path}' was not found.")
    except KeyError as e:
        print(f"Error: A required column is missing from the CSV file. {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def execute_code(code_to_run: str) -> int:
    """
    Executes a string of Python code in an isolated process.

    Args:
        code_to_run (str): The Python code to execute.

    Returns:
        int: Returns 1 for successful execution, 0 for any crash (e.g., an assertion failure).
    """
    def worker():
        try:
            exec(code_to_run, {})
        except Exception:
            sys.exit(1)
        sys.exit(0)
    process = multiprocessing.Process(target=worker)
    process.start()
    process.join()
    return 1 if process.exitcode == 0 else 0