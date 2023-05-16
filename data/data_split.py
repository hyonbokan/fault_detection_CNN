import os
import shutil

# Specify the dataset directory, output directory, and number of clients
dataset_directory = "../../data_seismic/cl/train/"
output_directory = "../../data_seismic/fl/"
num_clients = 2

def split_datasets(dataset_dir, output_dir, num_clients):
    
    for i in range(1, num_clients + 1):
        client_dir = os.path.join(output_dir, f"client{i}/train")
        os.makedirs(client_dir, exist_ok=True)
        os.makedirs(os.path.join(client_dir, "seis"), exist_ok=True)
        os.makedirs(os.path.join(client_dir, "fault"), exist_ok=True)

    seis_dir = os.path.join(dataset_dir, "seis")
    seis_files = os.listdir(seis_dir)

    fault_dir = os.path.join(dataset_dir, "fault")
    fault_files = os.listdir(fault_dir)

    assert len(seis_files) == len(fault_files), "Seismic and fault files do not match."

    # Calculate the number of files per client
    files_per_client = len(seis_files) // num_clients

    # Copy files to client directories
    for i in range(num_clients):
        client_dir = os.path.join(output_dir, f"client{i+1}/train")
        start_index = i * files_per_client
        end_index = (i + 1) * files_per_client

        # Copy seismic files
        for file in seis_files[start_index:end_index]:
            file_path = os.path.join(seis_dir, file)
            shutil.copy(file_path, os.path.join(client_dir, "seis"))

        # Copy fault files 
        for file in fault_files[start_index:end_index]:
            file_path = os.path.join(fault_dir, file)
            shutil.copy(file_path, os.path.join(client_dir, "fault"))

    # Check if file names match in each client directory
    for i in range(1, num_clients + 1):
        client_dir = os.path.join(output_dir, f"client{i}/train")
        client_seis_files = os.listdir(os.path.join(client_dir, "seis"))
        client_fault_files = os.listdir(os.path.join(client_dir, "fault"))

        assert set(client_seis_files) == set(client_fault_files), f"File names do not match in client{i} directory."


# Call the split_datasets function
split_datasets(dataset_directory, output_directory, num_clients)
print("Finished")