import os
import shutil

# Specify the directory and number of clients
directory = '/path/to/directory'  # Replace with your directory path
num_clients = 2  # Replace with the desired number of clients

def split_files(directory, num_clients):
    # Create client directories if they don't exist
    for i in range(1, num_clients + 1):
        client_dir = os.path.join(directory, f"client{i}")
        os.makedirs(client_dir, exist_ok=True)
    
    # Get the list of files in the directory
    files = os.listdir(directory)
    # Calc number of files per client
    files_per_client = len(files) // num_clients
    
    # Copy files to client directories
    for i in range(num_clients):
        client_dir = os.path.join(directory, f"client{i+1}")
        start_index = i * files_per_client
        end_index = (i + 1) * files_per_client
        
        # Copy files for the current client
        for file in files[start_index:end_index]:
            file_path = os.path.join(directory, file)
            shutil.copy(file_path, client_dir)


split_files(directory, num_clients)
