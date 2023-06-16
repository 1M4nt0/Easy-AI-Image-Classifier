import os
import shutil

def count_folders(path):
    folder_count = 0

    # Iterate over all items in the given path
    for item in os.listdir(path):
        item_path = os.path.join(path, item)

        # Check if the item is a directory/folder
        if os.path.isdir(item_path):
            folder_count += 1

    return folder_count

def remove_folder(path):
    if os.path.exists(path):
        shutil.rmtree(path)
        print(f"Folder '{path}' has been removed.")
    else:
        print(f"Folder '{path}' does not exist.")