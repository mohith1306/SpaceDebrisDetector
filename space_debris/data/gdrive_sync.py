"""
Google Drive Sync Module
Downloads 'Shared with me' folders to local cache for training.
Uses PyDrive2 with OAuth (first run opens browser for auth).
"""
import os
import json
from pathlib import Path
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from tqdm import tqdm


class GDriveSync:
    def __init__(self, cache_dir="E:/Capstone/gdrive_cache", credentials_file="credentials.json"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.credentials_file = Path(credentials_file)
        self.drive = None
        self._authenticate()

    def _authenticate(self):
        gauth = GoogleAuth()
        
        # Try to load saved credentials
        if self.credentials_file.exists():
            gauth.LoadCredentialsFile(str(self.credentials_file))
        
        if gauth.credentials is None:
            # First run: open browser for OAuth
            print("🔐 First run: Opening browser for Google Drive authentication...")
            gauth.LocalWebserverAuth()
        elif gauth.access_token_expired:
            print("🔄 Refreshing expired token...")
            gauth.Refresh()
        else:
            print("✅ Using saved credentials")
            gauth.Authorize()
        
        # Save credentials for next run
        gauth.SaveCredentialsFile(str(self.credentials_file))
        self.drive = GoogleDrive(gauth)
        print("✅ Google Drive authenticated")

    def find_folder(self, folder_name, parent_id=None):
        """Find folder by name, optionally under a parent."""
        query = f"title='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
        if parent_id:
            query += f" and '{parent_id}' in parents"
        
        file_list = self.drive.ListFile({'q': query}).GetList()
        return file_list[0] if file_list else None

    def find_shared_with_me_folder(self, folder_name):
        """Find folder in 'Shared with me'."""
        # Search all accessible files for the folder name
        query = f"title='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false and sharedWithMe=true"
        file_list = self.drive.ListFile({'q': query}).GetList()
        return file_list[0] if file_list else None

    def list_folder_contents(self, folder_id):
        """List all files/folders in a folder."""
        query = f"'{folder_id}' in parents and trashed=false"
        file_list = self.drive.ListFile({'q': query}).GetList()
        return file_list

    def download_file(self, file_obj, local_path):
        """Download a single file with progress."""
        if local_path.exists():
            # Check if file size matches (skip if same)
            try:
                remote_size = int(file_obj.get('fileSize', 0))
                local_size = local_path.stat().st_size
                if remote_size == local_size:
                    return True  # Already downloaded
            except:
                pass
        
        local_path.parent.mkdir(parents=True, exist_ok=True)
        file_obj.GetContentFile(str(local_path))
        return True

    def sync_folder(self, folder_id, local_dir, prefix=""):
        """Recursively sync a Google Drive folder to local directory."""
        local_dir = Path(local_dir)
        local_dir.mkdir(parents=True, exist_ok=True)
        
        contents = self.list_folder_contents(folder_id)
        
        files = [f for f in contents if f['mimeType'] != 'application/vnd.google-apps.folder']
        folders = [f for f in contents if f['mimeType'] == 'application/vnd.google-apps.folder']
        
        # Download files
        for file_obj in tqdm(files, desc=f"{prefix}Downloading files", leave=False):
            local_file = local_dir / file_obj['title']
            try:
                self.download_file(file_obj, local_file)
            except Exception as e:
                print(f"⚠️ Failed to download {file_obj['title']}: {e}")
        
        # Recurse into subfolders
        for folder in folders:
            subdir = local_dir / folder['title']
            self.sync_folder(folder['id'], subdir, prefix + "  ")

    def sync_capstone_dataset(self, folder_name="Capstone", local_name="capstone_data"):
        """
        Find and sync the Capstone folder from 'Shared with me' to local cache.
        Returns local path to the synced data.
        """
        print(f"🔍 Searching for '{folder_name}' in Shared with me...")
        
        # Try "Shared with me" first
        folder = self.find_shared_with_me_folder(folder_name)
        
        if not folder:
            print(f"⚠️ Not found in 'Shared with me', searching all accessible drives...")
            folder = self.find_folder(folder_name)
        
        if not folder:
            raise FileNotFoundError(f"Folder '{folder_name}' not found in Google Drive")
        
        print(f"✅ Found: {folder['title']} (ID: {folder['id']})")
        
        local_path = self.cache_dir / local_name
        print(f"📥 Syncing to {local_path}...")
        self.sync_folder(folder['id'], local_path)
        
        print(f"✅ Sync complete: {local_path}")
        return local_path


def sync_capstone_data(cache_dir="E:/Capstone/gdrive_cache", folder_name="Capstone"):
    """Convenience function to sync Capstone dataset."""
    sync = GDriveSync(cache_dir=cache_dir)
    return sync.sync_capstone_dataset(folder_name=folder_name)


if __name__ == "__main__":
    import sys
    folder = sys.argv[1] if len(sys.argv) > 1 else "Capstone"
    path = sync_capstone_data(folder_name=folder)
    print(f"\n📁 Dataset ready at: {path}")