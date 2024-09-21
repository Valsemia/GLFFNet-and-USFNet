import os

def list_files_in_directory(directory_path, output_file):
    """
    List all files in the given directory and write their names and extensions to a text file.

    Args:
    - directory_path (str): The path to the directory to list files from.
    - output_file (str): The path to the output text file.
    """
    with open(output_file, 'w') as file:
        for filename in os.listdir(directory_path):
            # Split the filename into name and extension
            name, extension = os.path.splitext(filename)
            # Write the filename and extension to the output file
            file.write(f"{name}{extension}\n")

# Example usage
directory_path = r'E:\Code\Remove\Remove +迁移\SG-ShadowNet-main\Dataset\ISTD_Dataset\test\test_A'
output_file = 'output.txt'
list_files_in_directory(directory_path, output_file)
