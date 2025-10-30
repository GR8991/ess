import streamlit as st
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
import google.auth.transport.requests
from google.oauth2 import id_token

CLIENT_SECRETS_FILE = 'credentials.json'
SCOPES = ['https://www.googleapis.com/auth/drive.metadata.readonly', 'openid', 'https://www.googleapis.com/auth/userinfo.email']

AUTHORIZED_USERS = ['user1@gmail.com', 'user2@gmail.com']  # Add your team email addresses here

if 'credentials' not in st.session_state:
    st.session_state['credentials'] = None
if 'email' not in st.session_state:
    st.session_state['email'] = None

def login():
    flow = Flow.from_client_secrets_file(
        CLIENT_SECRETS_FILE,
        scopes=SCOPES,
        redirect_uri='http://localhost:8501'  # Update with your deployed URL
    )
    auth_url, _ = flow.authorization_url(prompt='consent', include_granted_scopes='true')
    st.write(f"[Click here to authenticate]({auth_url})")

    code = st.text_input('Enter the authorization code here:')
    if code:
        flow.fetch_token(code=code)
        creds = flow.credentials
        st.session_state['credentials'] = creds

        # Verify the ID token to get user info
        request = google.auth.transport.requests.Request()
        id_info = id_token.verify_oauth2_token(
            creds.id_token,
            request,
            audience=flow.client_config['client_id']
        )
        email = id_info.get('email')
        st.session_state['email'] = email

        st.experimental_rerun()

def list_drive_files():
    email = st.session_state['email']
    if email not in AUTHORIZED_USERS:
        st.error("Access Denied: Your account is not authorized to view these folders.")
        return

    creds = st.session_state['credentials']
    service = build('drive', 'v3', credentials=creds)
    results = service.files().list(q="mimeType='application/vnd.google-apps.folder'",
                                   pageSize=20, fields="files(id, name)").execute()
    folders = results.get('files', [])
    if not folders:
        st.write('No folders found.')
    else:
        st.write(f'Welcome {email}, here are your accessible folders:')
        for folder in folders:
            st.write(f"- {folder['name']} (ID: {folder['id']})")

st.title('Google Drive Folder Viewer with Restricted Access')

if st.session_state['credentials'] is None:
    login()
else:
    list_drive_files()
    if st.button('Logout'):
        st.session_state['credentials'] = None
        st.session_state['email'] = None
        st.experimental_rerun()
