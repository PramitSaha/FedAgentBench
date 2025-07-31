from langchain.tools import tool
import os
from langchain.tools.shell.tool import ShellTool
import shutil
import re
import csv
import json
from typing import List, Dict, Tuple
import json
import torch
from tqdm import tqdm
from torchvision.datasets import ImageFolder
from selfclean import SelfClean
from selfclean.cleaner.selfclean import PretrainingType, DINO_STANDARD_HYPERPARAMETERS
import copy
import pandas as pd
import torchvision.transforms as T
import selfclean.core.src.utils.utils as sc_utils
from PIL import Image

import selfclean
print(selfclean.__file__)
import ssl
from selfclean.core.src.pkg.embedder import Embedder
from types import SimpleNamespace
import copy
from types import SimpleNamespace
from typing import Tuple
from torchvision.datasets import ImageFolder
from torchvision import transforms
from selfclean import SelfClean
from selfclean.cleaner.selfclean import PretrainingType, DINO_STANDARD_HYPERPARAMETERS
from selfclean.core.src.pkg import embedder as sc_utils


def resize_images_in_folder(root_dir: str, size=(224, 224)):
    """
    Resize all images in the given directory (and subfolders) in-place.

    Args:
        root_dir (str): Path to the root folder containing class subfolders with images.
        size (tuple): Desired output image size (width, height).
    """
    for class_name in os.listdir(root_dir):
        class_path = os.path.join(root_dir, class_name)
        if os.path.isdir(class_path):
            for filename in tqdm(os.listdir(class_path), desc=f"Resizing {class_name}"):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    image_path = os.path.join(class_path, filename)
                    try:
                        img = Image.open(image_path).convert('RGB')
                        img = img.resize(size, Image.BICUBIC)
                        img.save(image_path)  # overwrite original
                    except Exception as e:
                        print(f"❌ Failed to process {image_path}: {e}")


# Monkeypatch: make init_distributed_mode do nothing
def dummy_init_distributed_mode():
    pass



def is_code_file(file_path):

    code_extensions = ['.py', '.java', '.cpp', '.c', '.js', '.html', '.css', '.php', '.go', '.rb', '.sh', '.json']
    return any(file_path.endswith(ext) for ext in code_extensions)

def should_skip_directory(dir_path, file_count_threshold=1000):
    """判断是否应该跳过该目录（基于文件数量）"""
    try:
        file_count = sum(1 for entry in os.scandir(dir_path) if entry.is_file())
        return file_count > file_count_threshold
    except (PermissionError, FileNotFoundError):
        # 如果无法访问目录，跳过
        return True

@tool
def traverse_dirs(directory: str) -> str:
    """
    Recursively get paths of all code files in a directory and return them as a single string.
    
    Args:
        directory: Directory path to scan

    Returns:
        str: A string containing paths of all code files, separated by newlines
    """
    print(f"Running traverse_dirs tool under {directory}...")
    code_files = []
    file_count_threshold = 10
    for root, dirs, files in os.walk(directory):
        if should_skip_directory(root, file_count_threshold):
            continue
        
        for file in sorted(files):
            file_path = os.path.join(root, file)
            if is_code_file(file_path):
                code_files.append(file_path)
        
        dirs.sort()
    

    code_files_str = '\n'.join(code_files)
    # print(code_files_str)
    return code_files_str

@tool
def read_files(file_paths: list) -> dict:
# def read_files(file_paths: List[str]) -> Dict[str, str]:

    """
    Read file contents and return as dictionary
    
    Args:
        file_paths: List of file paths to read

    Returns:
        dict: Dictionary with {file_path: file_content} format
    """
    # print(f"Running read_files tool to read {file_paths}...")
    file_contents = {}
    
    for file_path in file_paths:
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                file_contents[file_path] = content
        except (UnicodeDecodeError, PermissionError, FileNotFoundError) as e:
            print(f"Cannot read file {file_path}: {e}")
            file_contents[file_path] = None
    
    return file_contents

@tool
def copy_files(file_mapping: dict) -> str:
# def copy_files(file_mapping: Dict[str, str]) -> str:
    """
    Copy multiple files from source paths to destination paths.

    Args:
        file_mapping (dict): A dictionary where keys are source file paths and values are destination file paths.
            Example: {
                "/path/to/source1.txt": "/path/to/destination1.txt",
                "/path/to/source2.txt": "/path/to/destination2.txt"
            }

    Returns:
        str: A message indicating the result of the operation.
    """
    print(f"Running copy_files tool to copy {file_mapping}...") 
    results = []
    for src, dest in file_mapping.items():
        try:
            # 检查源文件是否存在
            if not os.path.exists(src):
                results.append(f"源文件 {src} 不存在。")
                continue

            # 检查目标路径是否存在，如果不存在则创建
            dest_directory = os.path.dirname(dest)
            if not os.path.exists(dest_directory):
                os.makedirs(dest_directory)

            # 复制文件
            shutil.copy2(src, dest)
            results.append(f"文件 {src} 已成功复制到 {dest}")

        except Exception as e:
            results.append(f"复制文件 {src} 时出错: {e}")

    # 返回所有任务的结果
    return "\n".join(results)

@tool
def copy_directory(src_dir: str, dest_dir: str) -> str:
    """
    Copy source directory and its contents to destination directory, creating a new subdirectory
    with the same name as the source directory.
    
    Args:
        src_dir: Source directory path (e.g., '/path/to/source/folder_name')
        dest_dir: Parent destination directory path (e.g., '/path/to/dest')
                 A new subdirectory named 'folder_name' will be created here

    Returns:
        str: Operation result message

    Example:
        If src_dir is '/path/to/source/folder_name' and dest_dir is '/path/to/dest',
        the contents will be copied to '/path/to/dest/folder_name'
    """
    print(f"Running copy_directory tool to copy from {src_dir} to {dest_dir}...")
    try:
        # Check if source directory exists
        if not os.path.exists(src_dir):
            return f"Source directory {src_dir} does not exist"
            
        # Get the source directory name
        src_name = os.path.basename(src_dir.rstrip('/'))
        target_dir = os.path.join(dest_dir, src_name)
        
        # If destination directory exists, remove it first
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)
            
        # Copy the entire directory tree
        shutil.copytree(src_dir, target_dir)
        return f"Directory {src_dir} has been successfully copied to {target_dir}"
        
    except Exception as e:
        return f"Error copying directory: {str(e)}"

@tool
def write_file(content: str, file_path: str) -> None:
    """
    Write a given string of code to a specified file.

    This function creates the necessary directories for the file (if they don't exist), 
    writes the content to the file, and handles any errors that may occur during the process.

    Args:
        content (str): The code or text you want to write into the file.
        file_path (str): The full path (including filename) where the content will be saved.

    Example:
        write_file('print("Hello World")', 'scripts/hello.py')
    """
    print(f"Running write_file tool to write {file_path}...")
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(content)
        
        print(f"File successfully written to: {file_path}")
    except Exception as e:
        print(f"Error writing file: {e}")

@tool
def edit_file(new_content: str, file_path: str) -> None:
    """
    Completely overwrite a file with new content. The original file content will be replaced entirely.
    
    Args:
        new_content: Complete content to replace the existing file content. This should be the entire
                    desired content of the file after editing, not just the changes.
        file_path: Path of the file to edit

    Note:
        This function performs a complete overwrite operation. The original content will be lost.
        You must provide the complete desired final content, including both modified and unmodified parts.
    """
    print(f"Running edit_file tool to edit {file_path}...")
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(new_content)
        
        print(f"File {file_path} successfully edited.")
    except Exception as e:
        print(f"Error editing file: {e}")   

@tool
def run_script(command: str) -> str:
    """
    Execute shell command
    
    Args:
        command: Shell command to execute

    Returns:
        str: Command execution result
    """
    cmd_base, script_path = command.strip().split(maxsplit=1)

    # Blindly quote the path
    script_path = f'"{script_path}"'

    # Rebuild the final command
    fixed_command = f"{cmd_base} {script_path}"

    print(f"Executing fixed command: {fixed_command}")
    print("Running run_script tool...")
    shell_tool = ShellTool()
    result = shell_tool.run({
        "commands": [fixed_command]
    })
    return result

import subprocess

# @tool
# def run_script(command: str) -> str:
#     """
#     Execute a shell command properly using subprocess.
    
#     Args:
#         command: Shell command to execute (e.g., "python /path/to/script.py")
    
#     Returns:
#         str: Execution result message.
#     """
#     print(f"Running run_script: {command}")
#     try:
#         # Actually run the command
#         result = subprocess.run(
#             command,
#             shell=True,
#             check=True,
#             stdout=subprocess.PIPE,
#             stderr=subprocess.PIPE,
#             text=True
#         )

#         # Print outputs
#         if result.stdout:
#             print("STDOUT:\n", result.stdout)
#         if result.stderr:
#             print("TDERR:\n", result.stderr)

#         return "Script executed successfully."

#     except subprocess.CalledProcessError as e:
#         print("Script execution failed!")
#         print(e.stderr)
#         return f"Script execution failed:\n{e.stderr}"



# @tool("list_files_in_second_level", 
#       description="Traverse the root directory and process each second-level entry. " 
#                   "For a file entry, record the file directly; for a directory entry, recursively "
#                   "collect all files (without depth limit), sort them naturally based on their relative paths, "
#                   "and return only the first max_files file paths along with the total file count. " 
#                   "Returns a dictionary with an 'entries' key containing the details.")

def natural_sort_key(s):
    """
    Generate a key for natural sorting.

    This function splits the string into numeric and non-numeric parts so that,
    for example, "file2" is sorted before "file10".

    Parameters:
      s (str): The string to generate a sorting key for.

    Returns:
      list: A list containing integers and lower-case string parts to be used as a sort key.
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def get_second_level_entries(root_dir):
    """
    Retrieve all second-level entries (files and directories) under the specified root directory,
    and sort them so that directories come first, followed by files. Both are sorted by natural order.

    Parameters:
      root_dir (str): The root directory path.

    Returns:
      list: A list of os.DirEntry objects representing the second-level entries.
            Returns an empty list if an error occurs while scanning.
    """
    try:
        entries = list(os.scandir(root_dir))
    except Exception as e:
        print(f"Error scanning {root_dir}: {e}")
        return []
    
    # Sort entries: directories first, then files; both sorted naturally.
    entries.sort(key=lambda e: (not e.is_dir(), natural_sort_key(e.name)))
    return entries

def collect_all_files_from_directory(directory):
    """
    Recursively traverse the given directory and collect all file paths.

    The function sorts directories and files naturally at each level.
    It returns a list of tuples (relative_path, full_file_path) where:
      - relative_path is the file path relative to the input directory.
      - full_file_path is the absolute path to the file.

    Parameters:
      directory (str): The directory to traverse.

    Returns:
      list: A list of tuples for all files, sorted naturally by the relative path.
    """
    collected = []
    for root, dirs, files in os.walk(directory):
        dirs.sort(key=natural_sort_key)
        files.sort(key=natural_sort_key)
        for file in files:
            full_file_path = os.path.join(root, file)
            relative_path = os.path.relpath(full_file_path, start=directory)
            collected.append((relative_path, full_file_path))
    collected.sort(key=lambda tup: natural_sort_key(tup[0]))
    return collected

@tool
def list_files_in_second_level(root_directory: str) -> dict:
# def list_files_in_second_level(root_directory: str) -> Dict[str, List[Dict[str, object]]]:

    """
    Traverse all second-level entries under the given root directory and return a dictionary.
    For each entry, if it is a file, record it directly with a total file count of 1.
    If the entry is a directory, recursively gather all file paths and record the total file count,
    but only include the first 100 file paths.
    Parameters:
    root_directory: The path to the root directory.
    Returns:
    A dictionary with a key named "entries". The value is a list of dictionaries where each
    dictionary has:
        entry_name - the name of the entry,
        entry_path - the full path of the entry,
        total_files - the number of files under the entry (1 if it is a file),
        files - a list of file paths (up to 100 items for directories).
    """
    print(f"Running list_files_in_second_level tool under {root_directory}...")
    max_files = 10  # Limit to at most 100 file paths for directory entries

    results = []
    second_level_entries = get_second_level_entries(root_directory)
    
    for entry in second_level_entries:
        if entry.is_file():
            # Record the file entry directly.
            result_dict = {
                "entry_name": entry.name,
                "entry_path": entry.path,
                "total_files": 1,
                "files": [entry.path]
            }
            results.append(result_dict)
        elif entry.is_dir():
            # Recursively collect all files from the directory.
            collected_files = collect_all_files_from_directory(entry.path)
            total_file_count = len(collected_files)
            # Only include the first max_files file paths.
            top_files = [full_path for _, full_path in collected_files[:max_files]]
            result_dict = {
                "entry_name": entry.name,
                "entry_path": entry.path,
                "total_files": total_file_count,
                "files": top_files
            }
            results.append(result_dict)
    
    final_result = {"entries": results}
    print(final_result)
    return final_result

@tool
def preview_file_content(file_path: str) -> str:
    """
    Reads a CSV, JSON, or TXT file and returns a preview string.
    
    CSV: returns the first 5 rows (comma-separated) and total rows.
    JSON: returns the first 5 key-value pairs (if dict) or 5 elements (if list) and total count.
    TXT: returns the first 10,000 words and total word count.
    """
    print(f"Running preview_file_content tool for {file_path}...")
    if file_path.lower().endswith('.csv'):
        rows = []
        total_rows = 0
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                for row in reader:
                    total_rows += 1
                    if total_rows <= 5:
                        rows.append(row)
        except Exception as e:
            return f"Error reading CSV file: {e}"
        
        preview_str = "CSV File Preview:\n"
        for row in rows:
            preview_str += ", ".join(row) + "\n"
        preview_str += f"Total rows: {total_rows}"
        return preview_str
    
    # JSON file processing
    elif file_path.lower().endswith('.json'):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            return f"Error reading JSON file: {e}"
        
        if isinstance(data, dict):
            items = list(data.items())
            total_items = len(items)
            preview_items = items[:5]
            preview_str = "JSON File Preview (first 5 key-value pairs):\n"
            for key, value in preview_items:
                preview_str += f"{key}: {value}\n"
            preview_str += f"Total key-value pairs: {total_items}"
        elif isinstance(data, list):
            total_items = len(data)
            preview_items = data[:5]
            preview_str = "JSON File Preview (first 5 elements):\n"
            for item in preview_items:
                preview_str += f"{item}\n"
            preview_str += f"Total elements: {total_items}"
        else:
            preview_str = f"Unsupported JSON type: {type(data)}"
        return preview_str

    # TXT file processing: Read the first 10,000 words
    elif file_path.lower().endswith('.txt'):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            return f"Error reading TXT file: {e}"
        
        words = content.split()  # split text by any whitespace
        total_words = len(words)
        preview_words = words[:10000]
        preview_str = "TXT File Preview (first 10000 words):\n"
        preview_str += " ".join(preview_words)
        preview_str += f"\nTotal words: {total_words}"
        return "=== CSV Preview === \n" + preview_str

    else:
        return "Unsupported file type. Only CSV, JSON, and TXT files are supported."


# @tool
# def run_selfclean_on_dataset(image_folder_path: str, output_folder_path: str) -> Tuple[str, str, str]:
#     """
#     Run SelfClean on an image folder and generate CSVs for near duplicates, off-topic samples, and label errors.

#     Args:
#         image_folder_path (str): Path to the root folder containing the images organized by class folders.
#         output_folder_path (str): Path to save the output CSV files.

#     Returns:
#         Tuple[str, str, str]: Paths to the generated CSV files: (near_duplicates_csv, off_topic_samples_csv, label_errors_csv)
#     """
#     sc_utils.init_distributed_mode = dummy_init_distributed_mode

#     # Patch torch.load for compatibility with SelfClean
#     original_torch_load = torch.load
#     def patched_torch_load(*args, **kwargs):
#         kwargs["weights_only"] = False
#         return original_torch_load(*args, **kwargs)
#     torch.load = patched_torch_load

#     os.makedirs(output_folder_path, exist_ok=True)
#     resize_images_in_folder(image_folder_path, output_folder_path)

#     # Load dataset
#     print("🔄 Loading dataset with ImageFolder...")
#     print("output_folder_path", output_folder_path)

#     from torchvision import transforms

# # Define the transform: resize, tensorize, normalize
#     transform = transforms.Compose([
#     transforms.ToTensor(),            # <-- Critical to convert PIL to torch.Tensor
#     #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
# ])

# # Load dataset with transform
#     dataset = ImageFolder(root=output_folder_path, transform=transform)

# # # Now sample_img will be a tensor
# #     sample_img, _ = dataset[0]
# #     print("✅ Sample loaded image shape:", sample_img.shape)  # Now this will work

# # # Load dataset with transforms
# #     dataset = ImageFolder(root=output_folder_path)
# #     # print(dataset)

#     # Prepare parameters
#     parameters = copy.deepcopy(DINO_STANDARD_HYPERPARAMETERS)
#     parameters['model']['base_model'] = 'pretrained_imagenet_vit_tiny'

#     # Run SelfClean
#     print("🚀 Running SelfClean...")
#     selfclean = SelfClean(
#         plot_top_N=7,
#         auto_cleaning=False,
#     )

#     issues = selfclean.run_on_dataset(
#         dataset=copy.copy(dataset),
#         pretraining_type=PretrainingType.DINO,
#         epochs=10,
#         batch_size=16,
#         save_every_n_epochs=1,
#         dataset_name="skin_cancer",
#         work_dir=output_folder_path,
#     )

#     # Extract issues
#     df_near_duplicates = issues.get_issues("near_duplicates", return_as_df=True)
#     df_off_topic_samples = issues.get_issues("off_topic_samples", return_as_df=True)
#     df_label_errors = issues.get_issues("label_errors", return_as_df=True)

#     # Save CSVs
#     near_duplicates_csv = os.path.join(output_folder_path, "near_duplicates.csv")
#     off_topic_samples_csv = os.path.join(output_folder_path, "off_topic_samples.csv")
#     label_errors_csv = os.path.join(output_folder_path, "label_errors.csv")

#     df_near_duplicates.to_csv(near_duplicates_csv, index=False)
#     df_off_topic_samples.to_csv(off_topic_samples_csv, index=False)
#     df_label_errors.to_csv(label_errors_csv, index=False)

#     print(f"✅ Saved CSVs to {output_folder_path}")

    

# # === Config ===
#     json_path = "/path/to/your/file.json"  # <-- update this to your JSON file path
#     output_dir = "..."
#     os.makedirs(output_dir, exist_ok=True)

#     # === Load JSON ===
#     with open(json_path, 'r') as f:
#         data = json.load(f)

#     # === Copy files ===
#     for idx, (filepath, label) in enumerate(data.items()):
#         label = label.strip().lower()
#         new_name = f"{label}_{idx:04d}.png"
#         dest_path = os.path.join(output_dir, new_name)

#         if os.path.isfile(filepath):
#             shutil.copyfile(filepath, dest_path)
#         else:
#             print(f"❌ File not found: {filepath}")

#     print("✅ Done copying files with new names.")


#     return near_duplicates_csv, off_topic_samples_csv, label_errors_csv




@tool
def run_selfclean_on_dataset(image_folder_path: str) -> None: #-> Tuple[str, str, str]:
    """
    Run SelfClean on an image folder and generate CSVs for near duplicates, off-topic samples, and label errors.

    Args:
        image_folder_path (str): Path to the root folder containing the images organized by class folders.

    Returns:
        Tuple[str, str, str]: Paths to the generated CSV files.
    """
    sc_utils.init_distributed_mode = dummy_init_distributed_mode

    # Patch torch.load for compatibility with SelfClean
    original_torch_load = torch.load

    def patched_torch_load(*args, **kwargs):
        kwargs["weights_only"] = False
        return original_torch_load(*args, **kwargs)

    torch.load = patched_torch_load
    resize_images_in_folder(image_folder_path)

    print("🔄 Loading dataset with ImageFolder...")
    # transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),])
    dataset = ImageFolder(root=image_folder_path)#, transform=transform)

    parameters = copy.deepcopy(DINO_STANDARD_HYPERPARAMETERS)
    parameters['model']['base_model'] = 'pretrained_imagenet_vit_tiny'

    print("🚀 Running SelfClean...")
    selfclean = SelfClean(auto_cleaning=True)
    print("Selfclean loaded")

    def patched_load_pretrained(model_name=None, work_dir=None, **kwargs):
        print("🔁 Using locally downloaded DINO checkpoint")
        local_model_path = "....pth"
        model = sc_utils.Embedder.load_dino(ckp_path=local_model_path)
        dummy_config = SimpleNamespace(model_type="ViT")
        dummy_augment_fn = lambda x: x
        return model, dummy_config, dummy_augment_fn

    sc_utils.Embedder.load_pretrained = patched_load_pretrained
    image_folder = image_folder_path
    work_folder_path = {    "..."}.get(image_folder, None)


    issues = selfclean.run_on_dataset(
        dataset=copy.copy(dataset),
        pretraining_type=PretrainingType.DINO,
        epochs=10,
        batch_size=16,
        save_every_n_epochs=1,
        dataset_name="skin_cancer",
        work_dir=work_folder_path,
    )

    df_near_duplicates = issues.get_issues("near_duplicates", return_as_df=True)
    df_off_topic_samples = issues.get_issues("off_topic_samples", return_as_df=True)
    df_label_errors = issues.get_issues("label_errors", return_as_df=True)

    # near_duplicates_csv = os.path.join(image_folder_path, "near_duplicates.csv")
    # off_topic_samples_csv = os.path.join(image_folder_path, "off_topic_samples.csv")
    # label_errors_csv = os.path.join(image_folder_path, "label_errors.csv")

    # df_near_duplicates.to_csv(near_duplicates_csv, index=False)
    # df_off_topic_samples.to_csv(off_topic_samples_csv, index=False)
    # df_label_errors.to_csv(label_errors_csv, index=False)

    # print(f"✅ Saved CSVs to {image_folder_path}")

    # return near_duplicates_csv, off_topic_samples_csv, label_errors_csv
 
@tool
def organize_into_subfolder(root_directory: str, destination_directory: str) -> dict:
    """
    Reads a CSV file from root_directory, finds the labels column,
    and organizes images into subfolders based on label values in destination_directory.
    
    Assumes:
    - There is exactly one CSV file in root_directory.
    - The CSV contains at least two columns: one for image filenames and one for labels.
    - All image files are located in root_directory (not yet in subfolders).

    Returns:
        dict: Summary of number of images moved per class to destination_directory.
    """
    try:
        # Step 1: Locate CSV file
        csv_files = [f for f in os.listdir(root_directory) if f.endswith(".csv")]
        if len(csv_files) != 1:
            return {"status": "error", "message": "Expected exactly one CSV file in the directory."}
        
        csv_path = os.path.join(root_directory, csv_files[0])
        df = pd.read_csv(csv_path)
        # print("csv file Read!")

        # Step 2: Auto-detect label and filename columns
        potential_label_cols = [col for col in df.columns if "label" in col.lower()]
        potential_file_cols = [col for col in df.columns if "file" in col.lower() or "image" in col.lower() or "path" in col.lower()]
        # print("potential_label_cols", potential_label_cols)
        # print("potential_file_cols", potential_file_cols)
        
        if not potential_label_cols or not potential_file_cols:
            return {"status": "error", "message": "Couldn't auto-detect label or image path columns in CSV."}

        label_col = potential_label_cols[0]
        file_col = potential_file_cols[0]

        moved_count = {}

        # Step 3: Move files based on label
        for _, row in df.iterrows():
            label = str(row[label_col]).strip()
            filename = str(row[file_col]).strip()
            # src_path = os.path.join(root_directory, filename)
            src_path = filename
            # print(src_path)

            if not os.path.exists(src_path):
                continue  # skip missing files

            label_folder = os.path.join(destination_directory, label)
            # print("label_folder", label_folder)
            os.makedirs(label_folder, exist_ok=True)

            dst_path = os.path.join(label_folder, os.path.basename(filename))
            shutil.copy2(src_path, dst_path)

            moved_count[label] = moved_count.get(label, 0) + 1

        return {"status": "success", "moved": moved_count}

    except Exception as e:
        return {"status": "error", "message": str(e)}



@tool
def copy_folder(source_directory: str, destination_directory: str) -> dict:
    """
    Copies all contents (files and subfolders) from source_directory to destination_directory.
    
    Args:
        source_directory (str): Path to the source folder.
        destination_directory (str): Path to the target folder.

    Returns:
        dict: A dictionary with the copy status.
    """
    try:
        if not os.path.exists(source_directory):
            return {"status": "error", "message": f"Source folder does not exist: {source_directory}"}

        if not os.path.exists(destination_directory):
            os.makedirs(destination_directory)

        for item in os.listdir(source_directory):
            src_path = os.path.join(source_directory, item)
            dst_path = os.path.join(destination_directory, item)

            if os.path.isdir(src_path):
                shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
            else:
                shutil.copy2(src_path, dst_path)

        return {"status": "success", "message": f"Copied from {source_directory} to {destination_directory}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@tool
def remove_other_files(root_directory: str) -> dict:
    """
    Removes all non-image files from the specified directory and its subdirectories.

    Keeps files with extensions: .jpg, .jpeg, .png, .bmp, .tiff, .tif, .gif

    Args:
        root_directory (str): Path to the root directory.

    Returns:
        dict: Summary of deleted files count.
    """
    allowed_extensions = {
    '.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif', '.dcm',
    '.nii', '.nii.gz', '.mha', '.mhd', '.hdr', '.img', '.nrrd'
}
    removed_files = []

    for dirpath, _, filenames in os.walk(root_directory):
        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()
            if ext not in allowed_extensions:
                file_path = os.path.join(dirpath, filename)
                try:
                    os.remove(file_path)
                    removed_files.append(file_path)
                except Exception as e:
                    print(f"Error removing {file_path}: {e}")

    return {
        "status": "success",
        "removed_file_count": len(removed_files),
        "removed_files": removed_files
    }


@tool
def list_folders(root_directory: str) -> dict:
    """
    List subfolders in the given directory.
    """
    folders = [f for f in os.listdir(root_directory) if os.path.isdir(os.path.join(root_directory, f))]
    return {"folders": folders}

@tool
def make_folder(root_directory: str) -> dict:
    """
    Create a new folder at the given path.
    """
    try:
        os.makedirs(root_directory, exist_ok=True)
        return {"status": "success", "message": f"Created folder: {root_directory}"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@tool
def copy_images(src_folder: str, dst_folder: str) -> dict:
    """
    Copies all image files from the source folder (including subfolders) to the destination folder.

    Args:
        src_folder (str): Path to the source folder containing image files.
        dst_folder (str): Path to the destination folder where images will be copied.

    Returns:
        dict: Summary of copied images including total copied count and failed files.
    """
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif', '.dcm'}
    copied_files = []
    failed_files = []

    os.makedirs(dst_folder, exist_ok=True)

    for root, _, files in os.walk(src_folder):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in allowed_extensions:
                src_path = os.path.join(root, file)
                dst_path = os.path.join(dst_folder, file)
                
                try:
                    shutil.copy2(src_path, dst_path)
                    copied_files.append(file)
                except Exception as e:
                    failed_files.append((file, str(e)))

    return {
        "status": "success",
        "copied_count": len(copied_files),
        "failed_count": len(failed_files),
        "failed_files": failed_files
    }


import subprocess
from typing import Dict

@tool
def run_federated_method(project_directory: str, method_name: str) -> Dict:
    """
    Runs `python main.py method=<method_name>` in the specified directory using subprocess.

    Args:
        project_directory (str): Path to the directory where main.py exists.
        method_name (str): Name of the federated learning method (e.g., 'fedavg', 'fedala').

    Returns:
        dict: Output status and any captured stdout/stderr from the command.
    """
    try:
        result = subprocess.run(
            ["python", ".../main.py", f"method={method_name}"],
            cwd=project_directory,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # print(result.stdout)

        return {
            "status": "success" if result.returncode == 0 else "failed",
            "stdout": result.stdout,
            "stderr": result.stderr,
            "exit_code": result.returncode
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
