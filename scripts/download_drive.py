import io
import os
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

# pip install google-auth google-api-python-client


# Path to the service account credentials file
SERVICE_ACCOUNT_FILE = 'credentials.json'

# Define the scopes
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

def main():
    # Authenticate and construct the service
    credentials = service_account.Credentials.from_service_account_file(
        SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    drive_service = build('drive', 'v3', credentials=credentials)

    # Get the folder ID from the URL
    url = 'https://drive.google.com/drive/folders/12RsjlEtSlBldhoQVJapfC9lIFZwpnVk6' # mace train
    # url = 'https://drive.google.com/drive/folders/10atLHo2VJyTcb9JqL8SjHZIornPntUh-'  # mace val
    folder_id = url.split("/")[-1]

    # Create a directory to store the downloaded files
    download_dir = os.path.join(os.getcwd(), "downloaded_files")
    if not os.path.exists(download_dir):
        os.mkdir(download_dir)

    # Download the files
    page_token = None
    while True:
        response = drive_service.files().list(q="'{}' in parents".format(folder_id),
                                              spaces='drive',
                                              fields='nextPageToken, files(id, name)',
                                              pageToken=page_token).execute()
        files = response.get('files', [])
        for file in files:
            request = drive_service.files().get_media(fileId=file['id'])
            fh = io.FileIO(os.path.join(download_dir, file['name']), 'wb')
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
                print(f'Download {int(status.progress() * 100)}.')

        page_token = response.get('nextPageToken', None)
        if page_token is None:
            break

if __name__ == '__main__':
    main()
