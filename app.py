import os
import uuid
import datetime
from flask import Flask, request, jsonify, render_template, redirect, url_for, session, flash, make_response
from werkzeug.utils import secure_filename
from google.cloud import documentai_v1 as documentai
from google import genai
from dotenv import load_dotenv
from collections import Counter
import logging
from pathlib import Path
import json
import re
import base64
import bcrypt
# Additional imports for Firestore & Cloud Storage
from google.cloud import firestore
from google.cloud import storage
from google.oauth2 import service_account
# Additional imports for Excel generation
import openpyxl
from openpyxl.styles import Alignment, Font, Border, Side
from bs4 import BeautifulSoup
import io

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)  # Replace with a secure key in production

# Load environment variables
load_dotenv()
required_vars = [
    "DOC_AI_CREDENTIALS",
    "FIRESTORE_CREDENTIALS",
    "STORAGE_CREDENTIALS",
    "PROJECT_ID",
    "PROCESSOR_ID",
    "GEMINI_API_KEY",
    "BUCKET_NAME"
]

missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

# --- Service Account Credentials ---
# Document AI
doc_ai_credentials_path = os.getenv("DOC_AI_CREDENTIALS")
doc_ai_credentials = service_account.Credentials.from_service_account_file(doc_ai_credentials_path)

# Firestore
firestore_credentials_path = os.getenv("FIRESTORE_CREDENTIALS")
firestore_credentials = service_account.Credentials.from_service_account_file(firestore_credentials_path)

# Cloud Storage
storage_credentials_path = os.getenv("STORAGE_CREDENTIALS")
storage_credentials = service_account.Credentials.from_service_account_file(storage_credentials_path)

# --- Initialize Clients ---
PROJECT_ID = os.getenv("PROJECT_ID")
PROCESSOR_ID = os.getenv("PROCESSOR_ID")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
BUCKET_NAME = os.getenv("BUCKET_NAME")

try:
    # Document AI
    docai_client = documentai.DocumentProcessorServiceClient(credentials=doc_ai_credentials)
    processor_name = f"projects/{PROJECT_ID}/locations/us/processors/{PROCESSOR_ID}"

    # Gemini
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)

    # Firestore
    db = firestore.Client(project=PROJECT_ID, credentials=firestore_credentials)

    # Cloud Storage
    storage_client = storage.Client(project=PROJECT_ID, credentials=storage_credentials)
    bucket = storage_client.bucket(BUCKET_NAME)

except Exception as e:
    logger.error(f"Failed to initialize Google clients: {str(e)}")
    raise

# Allowed file types
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_document_bytes(file_bytes: bytes):
    """
    Process a circuit diagram image from in-memory bytes and return structured analysis.
    Returns: Dict with OCR outputs (raw_text, bounding_boxes, component_count, detailed_components)
    """
    try:
        raw_document = documentai.RawDocument(content=file_bytes, mime_type="image/png")
        request = documentai.ProcessRequest(name=processor_name, raw_document=raw_document)
        result = docai_client.process_document(request=request)
        ocr_result = result.document.text

        # Extract bounding boxes
        bounding_boxes = []
        for page in result.document.pages:
            for paragraph in page.paragraphs:
                bounding_poly = paragraph.layout.bounding_poly.vertices
                bounding_boxes.append({
                    "paragraph_text": ocr_result[
                        paragraph.layout.text_anchor.text_segments[0].start_index :
                        paragraph.layout.text_anchor.text_segments[0].end_index
                    ],
                    "paragraph_confidence": paragraph.layout.confidence,
                    "bounding_box": [
                        {"x": vertex.x, "y": vertex.y} for vertex in bounding_poly
                    ]
                })

        # Detect components
        lines = [line.strip().lower() for line in ocr_result.split('\n') if line.strip()]
        component_count = Counter()
        
        detailed_components = []
        for line_index, line in enumerate(lines):
            parts = line.split()
            line_details = {
                "line_number": line_index + 1,
                "content": line,
                "components": []
            }
            for part in parts:
                # Basic detection patterns
                if any(pattern in part for pattern in ['r', 'c', 'l', 'ic', 'u', 'q', 'd']):
                    component_count[part] += 1
                    line_details["components"].append(part)
                if any(unit in part for unit in ['ohm', 'f', 'h', 'v']):
                    component_count[part] += 1
                    line_details["components"].append(part)
            detailed_components.append(line_details)

        return {
            'raw_text': ocr_result,
            'bounding_boxes': bounding_boxes,
            'component_count': dict(component_count),
            'detailed_components': detailed_components
        }
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise

def clean_text(text: str) -> str:
    """
    Remove excessive asterisks and unnecessary formatting from text.
    """
    # Remove consecutive asterisks (e.g., ***, ****)
    text = re.sub(r'\*{2,}', '', text)
    text = re.sub(r'\'{2,}', '', text)
    # Remove leading/trailing asterisks
    text = re.sub(r'^\*+|\*+$', '', text, flags=re.MULTILINE)
    # Remove leading/trailing quotes
    text = re.sub(r"^'+|'+$", '', text, flags=re.MULTILINE)
    return text.strip()

def generate_structured_analysis(ocr_data: dict, file_bytes: bytes):
    """
    Use Gemini API to generate a structured analysis.
    ocr_data: The OCR data dictionary
    file_bytes: The in-memory bytes of the uploaded image
    """
    try:
        # Encode the image to Base64 for Gemini request
        base64_image = base64.b64encode(file_bytes).decode('utf-8')

        system_prompt = """You are an expert in analyzing electrical panel diagrams. Your task is to extract information from circuit diagrams and structure it into an HTML table for easy visualization. You will be given images of circuit diagrams along with their OCR data, which includes text, bounding boxes, and component information.

Your primary goal is to understand the structure of an electrical panel based on the components and connections shown, and categorize components into:

1.  **Incoming Section:** This is where the main power enters the panel. These are often components related to the source of power. Look for terms such as "cable termination", "incoming supply", or "main breaker". It can be at the top, bottom, or anywhere.

2.  **Bus Bar Section:** The bus bar is a common connection point where power is distributed. Usually this is represented by horizontal line. Look for terms related to a "bus bar," including its ratings (e.g. 1250A, 30). It is generally in between the incoming and outgoing section.

3.  **Outgoing Sections:** Power is routed from the bus bar to various outgoing circuits or loads. Look for components associated with circuits and load side, like relays, or transformers, cable connections, and breakers after bus bar.

**Instructions:**

1.  **Diagram Analysis:** Use the OCR data (text, bounding boxes) to identify components and group them into incoming, bus bar, and outgoing sections. Prioritize the understanding of component types and relationships from the image rather than strictly following a top-down approach. Use spatial relationships where available but don't rely just on the spatial positions.

2.  **Component Identification and Relationships:**
    *   For each component, extract the component's name, quantity, and specifications from the OCR data.
    *   Use the component names and associated specifications, their text content, along with the visual connections of lines in the images, to establish relationships and classify them into the correct section.
    *   Examples: Cables that are marked "cable termination" are usually in the incoming section. components on or below horizontal bus bar line are usually considered outgoing or bus bar components.
    * Identify components as cable, breaker, relay or transformers for better interpretation.

3.  **HTML Table Structure:** Construct an HTML table to represent this structure.
    *   The table should have four columns: "Section", "Name", "Quantity", and "Specifications".
     *   Use three sections for table rows: "Incoming Section," "Bus Bar Section," and "Outgoing Section."

4.  **Additional Notes:** If any component, specification, or quantity cannot be clearly identified and categorized into one of the three sections, include the component or the un-parsed data in a separate section called "Additional Notes" below the table output in the text format. Use simple bullet points for clarity. For instance use - before every point. If no component needs to be included in additional notes, don't include it.

5.  **Output:** Return the data in the format of the HTML Table. Followed by the Additional Notes if any is there.

        <div style='margin-bottom: 8px;'>
        <table style='border-collapse: collapse; width: 100%;'>
            <thead>
                <tr>
                    <th style='border: 1px solid black; padding: 8px;'>Section</th>
                    <th style='border: 1px solid black; padding: 8px;'>Name</th>
                    <th style='border: 1px solid black; padding: 8px;'>Quantity</th>
                    <th style='border: 1px solid black; padding: 8px;'>Specifications</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td style='border: 1px solid black; padding: 8px;'>Incoming Feeder Section</td>
                    <td style='border: 1px solid black; padding: 8px;'>MCB</td>
                    <td style='border: 1px solid black; padding: 8px;'>2</td>
                    <td style='border: 1px solid black; padding: 8px;'>2A</td>
                </tr>
                <tr>
                    <td style='border: 1px solid black; padding: 8px;'>Outgoing Feeder Section</td>
                    <td style='border: 1px solid black; padding: 8px;'>MCCB</td>
                    <td style='border: 1px solid black; padding: 8px;'>1</td>
                    <td style='border: 1px solid black; padding: 8px;'>630A, 4P</td>
                </tr>
            </tbody>
        </table>
    </div>
     - Additional Notes:
         - Example
        - Example


    Make the output clean, hierarchical, and easy to understand. Ensure the table is valid HTML, and the rows are dynamically populated based on the analysis."""

        # Build the Gemini request
        contents = json.dumps({
            'system_prompt': system_prompt,
            'ocr_data': {
                'raw_text': ocr_data['raw_text'],
                'bounding_boxes': ocr_data['bounding_boxes'],
                'component_count': ocr_data['component_count'],
                'detailed_components': ocr_data['detailed_components']
            },
            'image': base64_image
        })

        response = gemini_client.models.generate_content(
            model='gemini-2.0-flash-exp',
            contents=contents
        )
        cleaned_response = clean_text(response.text)

         # **Added Code to Remove Backticks**
        # Remove triple backticks and any language specifier like ```html
        cleaned_response = re.sub(r'^```(?:html)?\n?', '', cleaned_response)
        cleaned_response = re.sub(r'\n?```$', '', cleaned_response)

        return cleaned_response
    except Exception as e:
        logger.error(f"Error generating structured analysis: {str(e)}")
        raise

def generate_download_analysis(structured_analysis_html: str) -> dict:
    """
    Use Gemini API to generate a structured analysis JSON from the structured analysis HTML string.
    Args:
        structured_analysis_html (str): The structured analysis in HTML string.
    Returns:
        dict: The JSON data representing the structured analysis.
    """
    try:
        system_prompt = """You are an expert in analyzing electrical panel diagrams. Your task is to convert the given structured analysis HTML into a JSON format suitable for Excel generation. The structured analysis includes sections, component names, quantities, and specifications.

**Instructions:**

1. **Input Analysis:** You will receive an HTML table that categorizes electrical components into sections such as Incoming, Bus Bar, and Outgoing.

2. **JSON Structure:** Convert the HTML table into a JSON object with the following structure:

{
  "sections": [
    {
      "section": "Incoming Section",
      "components": [
        {
          "name": "MCB",
          "quantity": "2",
          "specifications": "2A"
        },
        ...
      ]
    },
    {
      "section": "Bus Bar Section",
      "components": [
        ...
      ]
    },
    {
      "section": "Outgoing Section",
      "components": [
        ...
      ]
    }
  ],
  "additional_notes": [
    "Note 1",
    "Note 2",
    ...
  ]
}

3. **Additional Notes:** If there are any components or data that do not fit into the predefined sections, include them in the "additional_notes" array.

4. **Output:** Ensure the JSON is clean, properly formatted, and can be directly parsed by Python.

**Example:**

Given the following HTML table:

<div style='margin-bottom: 8px;'>
<table style='border-collapse: collapse; width: 100%;'>
    <thead>
        <tr>
            <th style='border: 1px solid black; padding: 8px;'>Section</th>
            <th style='border: 1px solid black; padding: 8px;'>Name</th>
            <th style='border: 1px solid black; padding: 8px;'>Quantity</th>
            <th style='border: 1px solid black; padding: 8px;'>Specifications</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td style='border: 1px solid black; padding: 8px;'>Incoming Feeder Section</td>
            <td style='border: 1px solid black; padding: 8px;'>MCB</td>
            <td style='border: 1px solid black; padding: 8px;'>2</td>
            <td style='border: 1px solid black; padding: 8px;'>2A</td>
        </tr>
        <tr>
            <td style='border: 1px solid black; padding: 8px;'>Outgoing Feeder Section</td>
            <td style='border: 1px solid black; padding: 8px;'>MCCB</td>
            <td style='border: 1px solid black; padding: 8px;'>1</td>
            <td style='border: 1px solid black; padding: 8px;'>630A, 4P</td>
        </tr>
    </tbody>
</table>
</div>
 - Additional Notes:
     - Example
    - Example

The corresponding JSON should be:

{
  "sections": [
    {
      "section": "Incoming Feeder Section",
      "components": [
        {
          "name": "MCB",
          "quantity": "2",
          "specifications": "2A"
        }
      ]
    },
    {
      "section": "Outgoing Feeder Section",
      "components": [
        {
          "name": "MCCB",
          "quantity": "1",
          "specifications": "630A, 4P"
        }
      ]
    }
  ],
  "additional_notes": [
    "Example",
    "Example"
  ]
}

Ensure the JSON follows this structure accurately."""

        # Build the Gemini request
        contents = json.dumps({
            'system_prompt': system_prompt,
            'structured_analysis_html': structured_analysis_html
        })

        response = gemini_client.models.generate_content(
            model='gemini-2.0-flash-exp',
            contents=contents
        )
        
        # Extract JSON from response
        text = response.text.strip()
        
        # Remove any markdown code block indicators
        text = re.sub(r'^```\w*\n', '', text)
        text = re.sub(r'\n```$', '', text)
        
        # Try to find JSON content within the text
        json_match = re.search(r'(\{[\s\S]*\})', text)
        if json_match:
            text = json_match.group(1)
        
        # Debug logging
        logger.info(f"Attempting to parse text as JSON: {repr(text)}")
        try:
            json_data = json.loads(text)
            return json_data
        except ValueError as parse_error:
            logger.error(f"JSON parsing error. Response text: {text}")
            logger.error(f"Parse error: {str(parse_error)}")
            return {
                "sections": [],
                "additional_notes": [f"Error: Unable to parse Gemini response. Parse error: {str(parse_error)}"]
            }

    except Exception as e:
        logger.error(f"Error generating download analysis: {str(e)}")
        return {
            "sections": [],
            "additional_notes": [f"Error: An exception occurred during analysis: {str(e)}"]
        }

def hash_password(password: str) -> str:
    """Hash a plaintext password using bcrypt."""
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    """Verify a plaintext password against the hashed version."""
    return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Render and handle the login page."""
    if request.method == 'POST':
        username = request.form.get('username').strip()
        password = request.form.get('password').strip()

        if not username or not password:
            flash('Please enter both username and password.', 'error')
            return redirect(url_for('login'))

        try:
            user_ref = db.collection('users').document(username)
            user_doc = user_ref.get()
            if not user_doc.exists:
                flash('Invalid username or password.', 'error')
                return redirect(url_for('login'))

            user_data = user_doc.to_dict()
            hashed_password = user_data.get('password')

            if not verify_password(password, hashed_password):
                flash('Invalid username or password.', 'error')
                return redirect(url_for('login'))

            # Credentials are valid; set session
            session['username'] = username
            flash('Logged in successfully!', 'login_success')
            return redirect(url_for('index'))

        except Exception as e:
            logger.error(f"Login error: {str(e)}")
            flash('An error occurred during login. Please try again.', 'error')
            return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/logout')
def logout():
    """Log the user out."""
    session.pop('username', None)
    flash('Logged out successfully.', 'logout_success')
    return redirect(url_for('login'))

from functools import wraps

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'username' not in session:
            flash('Please log in to access this page.', 'error')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
@login_required
def index():
    """Render the upload page for authenticated users, 
       and automatically retrieve the last modified thread (if any)."""
    username = session['username']

    # Attempt to retrieve the latest (i.e., last modified) thread for this user
    threads_query = db.collection('users') \
                      .document(username) \
                      .collection('threads') \
                      .order_by('created_at', direction=firestore.Query.DESCENDING) \
                      .limit(1) \
                      .stream()
    
    last_thread = None
    for doc in threads_query:
        last_thread = doc.to_dict().get('thread_name')
        break

    return render_template('index.html', 
                           username=username, 
                           last_thread=last_thread)


@app.route('/getThreads', methods=['GET'])
@login_required
def get_threads():
    try:
        username = session['username']
        threads_ref = db.collection('users').document(username).collection('threads')\
                        .order_by('created_at', direction=firestore.Query.DESCENDING).stream()
        threads = []
        for doc in threads_ref:
            thread_data = doc.to_dict()
            threads.append({
                'thread_name': thread_data.get('thread_name'),
                'created_at': thread_data.get('created_at').isoformat() if thread_data.get('created_at') else None
            })
        return jsonify(threads)
    except Exception as e:
        logger.error(f"Error retrieving threads: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/getThreadImages', methods=['GET'])
@login_required
def get_thread_images():
    """
    Returns a list of images (with their outputs) for a given threadName.
    Example request: /getThreadImages?threadName=MyThread
    """
    try:
        username = session['username']
        thread_name = request.args.get('threadName')
        if not thread_name:
            return jsonify({'error': 'threadName query parameter is required'}), 400

        thread_ref = db.collection('users').document(username).collection('threads').document(thread_name)
        thread_doc = thread_ref.get()

        if not thread_doc.exists:
            return jsonify({'error': f'Thread "{thread_name}" not found'}), 404

        images_collection = thread_ref.collection('images').stream()
        images_list = []

        for img_doc in images_collection:
            img_data = img_doc.to_dict()
            gcs_path = img_data.get('gcs_path')
            
            # Build a public URL from 'gs://bucket/filename' -> 'https://storage.googleapis.com/bucket/filename'
            image_url = ''
            if gcs_path and gcs_path.startswith('gs://'):
                parts = gcs_path.replace('gs://', '').split('/', 1)
                if len(parts) == 2:
                    bucket_name, file_name = parts
                    image_url = f"https://storage.googleapis.com/{bucket_name}/{file_name}"

            images_list.append({
                'file_name': img_data.get('file_name'),
                'structured_analysis': img_data.get('structured_analysis'),
                'structured_analysis_json': img_data.get('structured_analysis_json'),
                'gcs_path': gcs_path,
                # Only new addition: 'image_url'
                'image_url': image_url
            })

        return jsonify(images_list), 200

    except Exception as e:
        logger.error(f"Error retrieving thread images: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/updateThreadName', methods=['POST'])
@login_required
def update_thread_name():
    """
    Update the name of an existing thread.

    Expects JSON payload:
    {
        "current_name": "Old Thread Name",
        "new_name": "New Thread Name"
    }
    """
    try:
        username = session['username']
        data = request.get_json()
        current_name = data.get('current_name')
        new_name = data.get('new_name')

        if not current_name or not new_name:
            return jsonify({'error': 'Both current_name and new_name are required.'}), 400

        # Check if the new thread name already exists to enforce uniqueness
        existing_thread_ref = db.collection('users').document(username).collection('threads').document(new_name)
        existing_thread_doc = existing_thread_ref.get()
        if existing_thread_doc.exists:
            return jsonify({'error': 'The new thread name already exists. Create a new thread with different name.'}), 400

        # Fetch the thread document
        threads_stream = db.collection('users').document(username).collection('threads')\
                           .where('thread_name', '==', current_name).stream()
        threads = list(threads_stream)

        if not threads:
            return jsonify({'error': 'Thread not found.'}), 404

        # Assuming thread names are unique; update the first match
        thread_doc = threads[0].reference
        thread_doc.update({'thread_name': new_name})

        logger.info(f"Thread name updated from '{current_name}' to '{new_name}' for user '{username}'.")

        return jsonify({'message': 'Thread name updated successfully.'}), 200

    except Exception as e:
        logger.error(f"Error updating thread name: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/checkThreadName', methods=['GET'])
@login_required
def check_thread_name():
    """
    Check if a thread name already exists in Firestore for the logged-in user.
    Example request: /checkThreadName?name=MyThread
    Response: { "exists": true/false }
    """
    try:
        username = session['username']
        thread_name = request.args.get('name', '').strip()
        if not thread_name:
            return jsonify({'exists': False}), 200

        thread_ref = db.collection('users').document(username).collection('threads').document(thread_name)
        thread_doc = thread_ref.get()

        # If doc exists, name is taken
        if thread_doc.exists:
            return jsonify({'exists': True}), 200
        return jsonify({'exists': False}), 200

    except Exception as e:
        logger.error(f"Error checking thread name: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/upload', methods=['POST'])
@login_required
def upload_file():
    """Handle file upload and processing."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    thread_name = request.form.get('thread_name')

    if not thread_name or thread_name.strip() == '':
        return jsonify({'error': 'Thread name is required and cannot be empty.'}), 400

    if file and allowed_file(file.filename):
        # 1) Secure the filename
        filename = secure_filename(file.filename)

        username = session['username']

        # 2) Upload the file to Cloud Storage
        blob = bucket.blob(f"{username}/{thread_name}/{filename}")
        blob.upload_from_file(file, content_type=file.content_type)

        # GCS path reference
        gcs_path = f"gs://{BUCKET_NAME}/{username}/{thread_name}/{filename}"

        # We need file bytes again for Document AI => re-seek or read the original file.
        file.seek(0)  # Reset file pointer to the beginning
        file_bytes = file.read()
        if not file_bytes:
            # If the pointer was consumed, download it from GCS in memory:
            file_bytes = blob.download_as_bytes()

        try:
            # 4) Document AI OCR
            ocr_data = process_document_bytes(file_bytes)

            # 5) Generate Structured Analysis (Gemini)
            structured_analysis = generate_structured_analysis(ocr_data, file_bytes)

            # 6) Store results in Firestore
            thread_ref = db.collection('users').document(username).collection('threads').document(thread_name)
            # Create (or update) the thread document
            thread_ref.set({
                'thread_name': thread_name,
                'created_at': firestore.SERVER_TIMESTAMP
            }, merge=True)

            image_id = str(uuid.uuid4())
            images_ref = thread_ref.collection('images').document(image_id)
            images_ref.set({
                'file_name': filename,
                'gcs_path': gcs_path,
                'ocr_data': ocr_data,  # < 1MB, so safe for Firestore
                'structured_analysis': structured_analysis,
                'uploaded_at': datetime.datetime.utcnow()
            })

            return jsonify({'structured_analysis': structured_analysis})

        except Exception as e:
            logger.error(f"Upload error: {str(e)}")
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'Invalid file format. Only PNG, JPG, JPEG allowed.'}), 400

@app.route('/downloadAnalysis', methods=['GET'])
@login_required
def download_analysis():
    """
    Generates an Excel file containing the combined structured analysis of all images in a thread and sends it as a download.
    """
    try:
        username = session['username']
        thread_name = request.args.get('threadName')
        if not thread_name:
            return jsonify({'error': 'threadName parameter is required'}), 400

        thread_ref = db.collection('users').document(username).collection('threads').document(thread_name)
        thread_doc = thread_ref.get()

        if not thread_doc.exists:
            return jsonify({'error': f'Thread "{thread_name}" not found'}), 404

        images_collection = thread_ref.collection('images').stream()
        excel_data = []
        all_additional_notes = []
        json_counter = 1  # To create unique JSON field names

        for img_doc in images_collection:
            img_data = img_doc.to_dict()
            structured_analysis = img_data.get('structured_analysis')
            gcs_path = img_data.get('gcs_path')

            if structured_analysis and gcs_path:
                # Download the file from GCS for processing, if gcs_path exists
                if gcs_path.startswith('gs://'):
                    parts = gcs_path.replace('gs://', '').split('/', 1)
                    if len(parts) == 2:
                        bucket_name, file_name = parts
                        blob = storage_client.bucket(bucket_name).blob(file_name)
                        file_bytes = blob.download_as_bytes()
                    else:
                        logger.error(f"Invalid GCS path: {gcs_path}")
                        continue  # If there is no GCS path, don't process the image

                # Generate download analysis (JSON) from structured_analysis string
                download_analysis_json = generate_download_analysis(structured_analysis)

                # Store the JSON in Firestore
                images_ref = thread_ref.collection('images').document(img_doc.id)
                images_ref.update({
                    f'structured_analysis_json_{json_counter}': download_analysis_json
                })
                json_counter += 1

                # Extract and append data for Excel
                for section in download_analysis_json.get('sections', []):
                    for component in section.get('components', []):
                        excel_data.append({
                            'Section': section.get('section'),
                            'Name': component.get('name'),
                            'Quantity': component.get('quantity'),
                            'Specifications': component.get('specifications')
                        })
                # Accumulate all additional notes in all_additional_notes
                all_additional_notes.extend(download_analysis_json.get('additional_notes', []))

        # Convert collected data to Excel
        excel_file = json_to_excel(excel_data, all_additional_notes)

        # Prepare Response for download
        return excel_file

    except Exception as e:
        logger.error(f"Error generating download analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500

def json_to_excel(excel_data, additional_notes):
    """
    Converts structured analysis to an Excel file using openpyxl.
    Groups data by image, with headers and yellow separators between complete section groups.
    
    Args:
        excel_data (list): List of dictionaries containing section data
        additional_notes (list): List of strings for additional notes, grouped by image
    """
    wb = openpyxl.Workbook()
    ws = wb.active
    
    headers = ["Section", "Name", "Quantity", "Specifications"]
    ws.append(headers)

    # Styles
    header_font = Font(bold=True, color="FFFFFF")
    header_fill = openpyxl.styles.PatternFill(start_color="000000", end_color="000000", fill_type="solid")
    yellow_fill = openpyxl.styles.PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid")
    image_header_font = Font(bold=True, size=12)
    header_alignment = Alignment(horizontal='center', vertical='center', wrap_text=True)

    # Style the main headers
    for cell in ws[1]:
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = header_alignment

    # Group data by image
    image_groups = {}
    current_image = 1
    current_section_group = []
    last_section = None

    for row_data in excel_data:
        section = row_data.get("Section", "")
        
        # If we encounter the Incoming Section and we already have data,
        # it means we're starting a new image
        if section == "Incoming Section" and last_section is not None and last_section != "Incoming Section":
            if current_section_group:
                image_groups[f"Image {current_image}"] = current_section_group
                current_image += 1
                current_section_group = []
        
        current_section_group.append(row_data)
        last_section = section

    # Add the last group
    if current_section_group:
        image_groups[f"Image {current_image}"] = current_section_group

    # Write data to Excel with proper grouping and separation
    row_num = 2  # Start after header row

    for image_name, image_data in image_groups.items():
        # Add image header
        ws.cell(row=row_num, column=1, value=image_name)
        ws.cell(row=row_num, column=1).font = image_header_font
        row_num += 1

        # Process sections for this image
        for row_data in image_data:
            # Add data row
            ws.cell(row=row_num, column=1, value=row_data.get("Section", ""))
            ws.cell(row=row_num, column=2, value=row_data.get("Name", ""))
            ws.cell(row=row_num, column=3, value=row_data.get("Quantity", ""))
            ws.cell(row=row_num, column=4, value=row_data.get("Specifications", ""))
            row_num += 1

        # Add yellow separator row after each image's data
        for col in range(1, len(headers) + 1):
            cell = ws.cell(row=row_num, column=col)
            cell.fill = yellow_fill
        row_num += 1

    # Add additional notes, grouped by image if possible
    if additional_notes:
        # Add separator before additional notes section
        for col in range(1, len(headers) + 1):
            cell = ws.cell(row=row_num, column=col)
            cell.fill = yellow_fill
        row_num += 1

        # Try to match notes with images if they're grouped
        if isinstance(additional_notes, dict):
            # If notes are grouped by image
            for image_name, notes in additional_notes.items():
                ws.cell(row=row_num, column=1, value=f"Additional Notes for {image_name}:")
                ws.cell(row=row_num, column=1).font = image_header_font
                row_num += 1
                
                for note in notes:
                    ws.cell(row=row_num, column=1, value=note)
                    row_num += 1

                # Add separator between note groups
                for col in range(1, len(headers) + 1):
                    cell = ws.cell(row=row_num, column=col)
                    cell.fill = yellow_fill
                row_num += 1
        else:
            # If notes are not grouped, add them all together
            ws.cell(row=row_num, column=1, value="Additional Notes:")
            ws.cell(row=row_num, column=1).font = image_header_font
            row_num += 1
            
            for note in additional_notes:
                ws.cell(row=row_num, column=1, value=note)
                row_num += 1

    # Add borders to all cells
    thin_border = Border(left=Side(style='thin'), 
                        right=Side(style='thin'), 
                        top=Side(style='thin'), 
                        bottom=Side(style='thin'))

    for row in ws.iter_rows(min_row=1, max_row=row_num-1):
        for cell in row:
            cell.border = thin_border

    # Autosize columns
    for column_cells in ws.columns:
        length = max(len(str(cell.value)) if cell.value else 0 for cell in column_cells)
        adjusted_width = (length + 2)
        column_letter = openpyxl.utils.get_column_letter(column_cells[0].column)
        ws.column_dimensions[column_letter].width = adjusted_width

    # Create output stream
    excel_stream = io.BytesIO()
    wb.save(excel_stream)
    excel_stream.seek(0)

    # Prepare response
    response = make_response(excel_stream.read())
    response.headers.set('Content-Type', 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    response.headers.set('Content-Disposition', 'attachment', filename='analysis.xlsx')
    return response

if __name__ == '__main__':
    app.run(debug=True)
