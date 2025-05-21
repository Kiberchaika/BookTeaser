import os
import time
import requests
from datetime import datetime
import msvcrt  # For Windows file locking check
from pathlib import Path

class FileUploader:
    def __init__(self, storage_path="storage", upload_list_file="uploaded_files.txt", server_url="http://81.94.158.96:7781"):
        self.storage_path = Path(storage_path)
        self.upload_list_file = Path(upload_list_file)
        self.server_url = server_url
        self.uploaded_files = set()
        
        # Create storage directory if it doesn't exist
        self.storage_path.mkdir(exist_ok=True)
        
        # Load previously uploaded files
        self.load_uploaded_files()

    def load_uploaded_files(self):
        """Load the list of previously uploaded files."""
        if self.upload_list_file.exists():
            with open(self.upload_list_file, 'r') as f:
                self.uploaded_files = set(line.strip() for line in f)

    def save_uploaded_file(self, filename):
        """Save uploaded filename to the list."""
        with open(self.upload_list_file, 'a') as f:
            f.write(f"{filename}\n")
        self.uploaded_files.add(filename)

    def is_file_locked(self, filepath):
        """Check if file is locked/opened by another process."""
        try:
            with open(filepath, 'rb') as f:
                try:
                    msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                    msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
                    return False
                except IOError:
                    return True
        except IOError:
            return True

    def get_files_sorted_by_modified_date(self):
        """Get list of files in storage directory sorted by modified date (oldest first)."""
        files = []
        for file in self.storage_path.glob('*'):
            if file.is_file():
                files.append((file, file.stat().st_mtime))
        return [f[0] for f in sorted(files, key=lambda x: x[1])]

    def upload_file(self, filepath):
        """Upload file to external server."""
        try:
            with open(filepath, 'rb') as f:
                files = {'video': f}
                response = requests.post(f"{self.server_url}/upload", files=files)
                if response.status_code == 200:
                    print(f"Successfully uploaded {filepath}")
                    self.save_uploaded_file(filepath.name)
                    return True
                else:
                    print(f"Failed to upload {filepath}. Status code: {response.status_code}")
                    return False
        except Exception as e:
            print(f"Error uploading {filepath}: {str(e)}")
            return False

    def run(self):
        """Main loop to check and upload files."""
        print(f"Starting file uploader. Monitoring directory: {self.storage_path}")
        print(f"Uploading to server: {self.server_url}")
        
        while True:
            try:
                files = self.get_files_sorted_by_modified_date()
                
                for file in files:
                    if file.name not in self.uploaded_files:
                        print(f"Found new file to upload: {file}")
                        
                        if not self.is_file_locked(file):
                            self.upload_file(file)
                        else:
                            print(f"File {file} is currently in use, skipping...")
                
                time.sleep(5)  # Wait for 5 seconds before next check
                
            except Exception as e:
                print(f"Error in main loop: {str(e)}")
                time.sleep(5)  # Wait before retrying

if __name__ == "__main__":
    uploader = FileUploader()
    uploader.run()
