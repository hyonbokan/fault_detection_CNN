import os
import shutil

directory = '/path/to/directory'  
NUM_CLIENTS = 2  # Replace with your client names


def split_files(directory, clients):
    for client in clients:
        client_dir = os.path.join(directory, client)
        os.makedirs(client_dir, exist_ok=True)
    
    files = os.listdir(directory)

    files_per_client = len(files) // NUM_CLIENTS
    
    # Copy files to client directories
    for i, client in range(NUM_CLIENTS):
        client_dir = os.path.join(directory, client)
        start_index = i * files_per_client
        end_index = (i + 1) * files_per_client
        
        # Copy files for the current client
        for file in files[start_index:end_index]:
            file_path = os.path.join(directory, file)
            shutil.copy(file_path, client_dir)


# Call the split_files function
split_files(directory, NUM_CLIENTS)
