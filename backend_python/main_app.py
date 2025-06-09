# full-stack/backend_python/main_app.py

from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS
import base64
import cv2
import numpy as np
import os
import shutil
import time
import urllib.parse
import requests
from dotenv import load_dotenv
import uuid
import json
from deepface import DeepFace
from passlib.context import CryptContext
from twilio.rest import Client
from ultralytics import YOLO # Ensure YOLO is imported
from mtcnn import MTCNN
import yolo_yamnet_analyzer

load_dotenv()

app = Flask(__name__)
CORS(app)

# --- CONFIGURATION ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_MODEL_PATH_CONFIG = os.path.join(BASE_DIR, "best.pt")
FACE_DATABASE_PATH = os.path.join(BASE_DIR, 'face_database')
# THIS IS THE CORRECTED LINE:
VIDEO_BASE_PATH_ON_SERVER = '/Users/veeshal/qwen_app/full-stack/advanced_security_system_with_face_authentication/backend_python/static'
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
QWEN_API_URL = "http://35.244.63.84:8000"
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

TWILIO_AUTH_TOKEN='YOUR_TWILIO_AUTH_TOKEN_HERE'  # Replace with your actual Twilio Auth Token
TWILIO_PHONE_NUMBER="+14155238886"  # Replace with your actual Twilio phone number
TWILIO_ACCOUNT_SID = 'YOUR_TWILIO_ACCOUNT_SID_HERE'  # Replace with your actual Twilio Account SID
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN) if all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN]) else None
COMMANDER_PHONE_NUMBER = ""

AUTHORIZED_ROLES_FOR_CONTROL_CENTER = ["Commander", "Lead Analyst"]
# --- HELPER FUNCTIONS ---

def determine_threat_level(qwen_analysis: dict) -> str:
    """Parses Qwen analysis to determine a threat level."""
    status = qwen_analysis.get("Security_Status", "").lower()
    activity = qwen_analysis.get("Detected_Activity", "").lower()

    if "high threat" in status or "explosion" in activity or "weapon" in activity:
        return "high"
    if "suspicious" in status or "potential threat" in status or "breach" in activity:
        return "medium"
    
    return "low"

def send_sms_alert(message: str):
    """Sends an SMS alert to the commander via Twilio."""
    if not twilio_client or not TWILIO_PHONE_NUMBER or not COMMANDER_PHONE_NUMBER:
        print("[SMS_ALERT] Twilio is not configured. Skipping SMS.")
        return False
    try:
        print(f"[SMS_ALERT] Sending alert to {COMMANDER_PHONE_NUMBER}: {message}")
        twilio_client.messages.create(
            body=message,
            from_=TWILIO_PHONE_NUMBER,
            to=COMMANDER_PHONE_NUMBER
        )
        return True
    except Exception as e:
        print(f"[SMS_ALERT] FAILED to send SMS: {e}")
        return False
    

def extract_evenly_spaced_frames_base64(video_path: str, num_frames: int) -> list:
    if not os.path.exists(video_path): return []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int) if total_frames >= num_frames else np.arange(0, total_frames).astype(int)
    base64_frames = []
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            success, buffer = cv2.imencode('.jpg', frame);
            if success: base64_frames.append(base64.b64encode(buffer.tobytes()).decode('utf-8'))
    cap.release(); return base64_frames

def call_qwen_api_stream(frames_package_b64: list, prompt: str):
    if not QWEN_API_URL:
        yield "Error: Qwen API URL is not configured on the server."
        return
    full_api_url = f"{QWEN_API_URL.rstrip('/')}/v1/video/describe"
    payload = {"images": frames_package_b64, "prompt": prompt, "max_new_tokens": 1500, "temperature": 0.2, "top_p": 0.9}
    print(f"[QWEN_STREAM] Sending request to {full_api_url}...")
    try:
        response = requests.post(full_api_url, headers={"Content-Type": "application/json"}, json=payload, stream=True, timeout=300)
        response.raise_for_status()
        print("[QWEN_STREAM] Connection established, streaming response...")
        for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
            yield chunk
    except requests.exceptions.RequestException as e:
        print(f"[QWEN_STREAM] Error: {e}")
        yield f"Error connecting to Qwen API: {e}"

def remove_representation_pkl():
    pkl_files = [f for f in os.listdir(FACE_DATABASE_PATH) if f.endswith('.pkl')]
    for pkl_file in pkl_files:
        try: os.remove(os.path.join(FACE_DATABASE_PATH, pkl_file))
        except Exception: pass

# --- ROUTES ---
@app.route('/api/faces', methods=['GET'])
def get_faces():
    if not os.path.exists(FACE_DATABASE_PATH): return jsonify([])
    people = []
    for person_name in [d for d in os.listdir(FACE_DATABASE_PATH) if os.path.isdir(os.path.join(FACE_DATABASE_PATH, d))]:
        person_dir = os.path.join(FACE_DATABASE_PATH, person_name); image_files = os.listdir(person_dir)
        front = sorted([f for f in image_files if f.startswith('front_')]); left = sorted([f for f in image_files if f.startswith('left_')]); right = sorted([f for f in image_files if f.startswith('right_')])
        base_url = request.host_url.rstrip('/'); people.append({'id': person_name, 'name': person_name.replace('_', ' '), 'addedDate': time.ctime(os.path.getctime(person_dir)), 'imageUrlsFront': [f"{base_url}/face_db_images/{person_name}/{urllib.parse.quote(f)}" for f in front], 'imageUrlsLeft': [f"{base_url}/face_db_images/{person_name}/{urllib.parse.quote(f)}" for f in left], 'imageUrlsRight': [f"{base_url}/face_db_images/{person_name}/{urllib.parse.quote(f)}" for f in right],})
    return jsonify(people)

@app.route('/api/faces', methods=['POST'])
def add_face():
    data = request.get_json()
    name = data.get('name')
    password = data.get('password') # NEW
    role = data.get('role') # Get the role from the request
    image_groups = {"front": data.get('image_urls_front', []), "left": data.get('image_urls_left', []), "right": data.get('image_urls_right', [])}
    
    # NEW: Validate password presence
    if not all([name, password, role]):
        return jsonify({'message': 'Name, password, and role are required.'}), 400
    
    person_folder_name = name.replace(' ', '_')
    person_dir = os.path.join(FACE_DATABASE_PATH, person_folder_name)
    os.makedirs(person_dir, exist_ok=True)
    
    try:
        # Save images (logic is the same)
        for angle, images in image_groups.items():
            for i, image_data_url in enumerate(images):
                _, encoded_data = image_data_url.split(",", 1); img = cv2.imdecode(np.frombuffer(base64.b64decode(encoded_data), np.uint8), cv2.IMREAD_COLOR)
                if img is not None: cv2.imwrite(os.path.join(person_dir, f"{angle}_{i+1}.jpg"), img)
        
        # --- NEW: Hash and save the password ---
        password_hash = pwd_context.hash(password)
        # Save the hash in a simple text file inside the person's folder
        with open(os.path.join(person_dir, 'p_hash.txt'), 'w') as f:
            f.write(password_hash)
        with open(os.path.join(person_dir, 'role.txt'), 'w') as f:
            f.write(role)

        remove_representation_pkl()
        return jsonify({'message': f'Operator {name} added with role {role}.'}), 201
    except Exception as e:
        if os.path.exists(person_dir): shutil.rmtree(person_dir)
        return jsonify({'message': f'Error processing request: {str(e)}'}), 500

@app.route('/api/analyze-restricted-zone', methods=['POST'])
def analyze_restricted_zone_route():
    data = request.get_json()
    relative_video_path = data.get('video_path')
    if not relative_video_path: return jsonify({'error': 'No video_path provided.'}), 400
    
    absolute_video_path = os.path.join(VIDEO_BASE_PATH_ON_SERVER, relative_video_path)
    if not os.path.exists(absolute_video_path): return jsonify({'error': 'Video not found.'}), 404

    try:
        yolo_model = YOLO(YOLO_MODEL_PATH_CONFIG)
        cap = cv2.VideoCapture(absolute_video_path)
        frame_pos = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / 2)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
        ret, frame = cap.read()
        cap.release()
        if not ret: return jsonify({'error': 'Could not read frame from video.'}), 500

        results = yolo_model(frame, classes=[0], verbose=False) # Class 0 is 'person' in standard YOLO models
        person_detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                person_detections.append(frame[y1:y2, x1:x2])

        if not person_detections:
            return jsonify({"threatLevel": "low", "summary": "No people detected in the control center."})

        detected_personnel = []
        unauthorized_personnel = []
        unknown_faces = 0
        
        for face_crop in person_detections:
            try:
                dfs = DeepFace.find(img_path=face_crop, db_path=FACE_DATABASE_PATH, model_name='Facenet', enforce_detection=True, silent=True)
                if dfs and isinstance(dfs, list) and len(dfs) > 0 and not dfs[0].empty:
                    identity_path = dfs[0].iloc[0]['identity']
                    person_folder_name = os.path.basename(os.path.dirname(identity_path))
                    person_name = person_folder_name.replace('_', ' ')
                    
                    role_path = os.path.join(FACE_DATABASE_PATH, person_folder_name, 'role.txt')
                    user_role = "No Role Assigned"
                    if os.path.exists(role_path):
                        with open(role_path, 'r') as f: user_role = f.read().strip()
                    
                    detected_personnel.append({"name": person_name, "role": user_role})
                    if user_role not in AUTHORIZED_ROLES_FOR_CONTROL_CENTER:
                        unauthorized_personnel.append({"name": person_name, "role": user_role})
                else:
                    unknown_faces += 1
            except ValueError:
                unknown_faces += 1
                continue

        summary = f"Detected Personnel: {[p['name'] for p in detected_personnel] or ['None']}. "
        threat_level = "low"
        
        if unknown_faces > 0:
            summary += f"WARNING: {unknown_faces} UNKNOWN individual(s) detected. "
            threat_level = "high"
        
        if unauthorized_personnel:
            summary += f"ALERT: UNAUTHORIZED personnel detected - {', '.join([p['name'] + ' (' + p['role'] + ')' for p in unauthorized_personnel])}."
            threat_level = "high" if threat_level == "high" else "medium"

        if threat_level == "low":
            summary = "All personnel detected in the control center are authorized."

        return jsonify({"threatLevel": threat_level, "summary": summary, "details": detected_personnel})

    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({'error': f'Error during restricted zone analysis: {str(e)}'}), 500
    
@app.route('/api/authenticate-password', methods=['POST'])
def authenticate_password_route():
    data = request.get_json()
    name = data.get('name')
    password = data.get('password')

    if not name or not password:
        return jsonify({'authenticated': False, 'message': 'Name and password are required.'}), 400

    person_folder_name = name.replace(' ', '_')
    hash_file_path = os.path.join(FACE_DATABASE_PATH, person_folder_name, 'p_hash.txt')

    if not os.path.exists(hash_file_path):
        return jsonify({'authenticated': False, 'message': 'Operator not found or has no password set.'}), 404
    
    try:
        with open(hash_file_path, 'r') as f:
            stored_hash = f.read().strip()
        
        if pwd_context.verify(password, stored_hash):
            print(f"[AUTH_SUCCESS] Password verified for operator: {name}")
            return jsonify({'authenticated': True, 'message': f'Welcome, Operator {name}!'})
        else:
            print(f"[AUTH_FAILURE] Invalid password for operator: {name}")
            return jsonify({'authenticated': False, 'message': 'Invalid password.'})
    except Exception as e:
        print(f"[AUTH_FATAL_ERROR] Password auth error: {e}")
        return jsonify({'authenticated': False, 'message': 'A server error occurred during password authentication.'}), 500
    
@app.route('/api/faces/<person_id>', methods=['DELETE'])
def delete_face(person_id):
    person_dir = os.path.join(FACE_DATABASE_PATH, person_id)
    if not os.path.exists(person_dir): return jsonify({'message': 'Not found.'}), 404
    try: shutil.rmtree(person_dir); remove_representation_pkl(); return jsonify({'message': 'Deleted.'})
    except Exception as e: return jsonify({'message': f'Error: {str(e)}'}), 500

@app.route('/face_db_images/<path:filename>')
def serve_face_image(filename): return send_from_directory(FACE_DATABASE_PATH, filename)

@app.route('/api/operators', methods=['GET'])
def get_operators():
    """
    Scans the face database and returns a list of all registered operator names.
    """
    if not os.path.exists(FACE_DATABASE_PATH):
        return jsonify([])
    try:
        operator_names = [d.replace('_', ' ') for d in os.listdir(FACE_DATABASE_PATH) if os.path.isdir(os.path.join(FACE_DATABASE_PATH, d))]
        return jsonify(sorted(operator_names))
    except Exception as e:
        return jsonify({'error': f'Could not read operators: {e}'}), 500
    

@app.route('/api/authenticate-operator', methods=['POST'])
def authenticate_operator_route():
    data = request.get_json()
    if not data or 'image_data_url' not in data: return jsonify({'authenticated': False, 'message': 'No image data provided.'}), 400
    if not os.path.exists(FACE_DATABASE_PATH) or not os.listdir(FACE_DATABASE_PATH):
        print(f"[AUTH_ERROR] Face database is empty or not found at {FACE_DATABASE_PATH}")
        return jsonify({'authenticated': False, 'message': 'Authentication system not configured. No faces in database.'}), 500
    try:
        header, encoded_data = data['image_data_url'].split(",", 1); image_bytes = base64.b64decode(encoded_data)
        nparr = np.frombuffer(image_bytes, np.uint8); captured_image_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if captured_image_cv is None: return jsonify({'authenticated': False, 'message': 'Invalid image format from client.'}), 400
        try:
            dfs = DeepFace.find(img_path=captured_image_cv, db_path=FACE_DATABASE_PATH, model_name='Facenet', enforce_detection=True, silent=True)
            if dfs and isinstance(dfs, list) and len(dfs) > 0 and not dfs[0].empty:
                identity_path = dfs[0].iloc[0]['identity']; person_name = os.path.basename(os.path.dirname(identity_path)).replace('_', ' ')
                print(f"[AUTH_SUCCESS] Operator recognized as: {person_name}")
                return jsonify({'authenticated': True, 'message': f'Welcome, Operator {person_name}!'})
            else:
                print(f"[AUTH_FAILURE] No matching face found in the database.")
                return jsonify({'authenticated': False, 'message': 'Face not recognized in the database.'})
        except ValueError as ve:
            print(f"[AUTH_ERROR] DeepFace ValueError: {ve}")
            if "Face could not be detected" in str(ve) or "cannot be empty" in str(ve): return jsonify({'authenticated': False, 'message': 'Could not detect a clear face. Please try again.'})
            else: return jsonify({'authenticated': False, 'message': 'Face recognition processing error.'})
    except Exception as e:
        print(f"[AUTH_FATAL_ERROR] An unexpected error occurred: {e}"); import traceback; traceback.print_exc()
        return jsonify({'authenticated': False, 'message': 'A critical server error occurred during authentication.'}), 500

@app.route('/videos/<path:filename>')
def serve_video(filename): return send_from_directory(os.path.join(VIDEO_BASE_PATH_ON_SERVER, 'videos'), filename)

@app.route('/api/qwen-direct-analysis', methods=['POST'])
def qwen_direct_analysis_route():
    data = request.get_json(); relative_video_path = data.get('video_path')
    if not relative_video_path: return jsonify({'error': 'No video_path provided.'}), 400
    absolute_video_path = os.path.join(VIDEO_BASE_PATH_ON_SERVER, relative_video_path)
    if not os.path.exists(absolute_video_path): return jsonify({'error': 'Video not found.'}), 404
    try:
        frames_b64 = extract_evenly_spaced_frames_base64(absolute_video_path, 8)
        if not frames_b64: return jsonify({'error': 'Could not extract frames.'}), 500
        structured_prompt = f""" You are an advanced AI surveillance analyst. Analyze the following sequence of surveillance images and provide a structured JSON response with the detected activity, objects, and security status.
Your response should strictly follow this format:
{{
"Detected_Activity": "Choose ONE of the following: 'Civilian with Weapon', 'Fence Breach Attempt', 'Explosion Event', 'Routine Soldier Patrol', 'Unarmed Civilian Presence'.",
"Detected_Objects": ["List all significant detected objects like 'person', 'soldier', 'civilian', 'AK-47', 'truck', 'fence'."],
"Number_of_Civilians": <integer_count>,
"Number_of_Soldiers": <integer_count>,
"Number_of_Civilians_Armed": <integer_count>,
"Security_Status": "Provide a brief assessment like 'High Threat', 'Suspicious Activity', 'Nominal', 'Potential Threat'.",
"Summary": "Provide a one-sentence summary of the event."
}}

Analyze the images and fill in the values.
"""
        def stream_raw_response():
            for chunk in call_qwen_api_stream(frames_b64, structured_prompt):
                yield chunk

        return Response(stream_raw_response(), mimetype='text/plain') # Return as plain text
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error: {str(e)}'}), 500
    
    
@app.route('/api/trigger-sms-alert', methods=['POST'])
def trigger_sms_alert_route():
    data = request.get_json()
    message = data.get('message')
    if not message:
        return jsonify({'error': 'Message is required.'}), 400
    
    success = send_sms_alert(message)
    if success:
        return jsonify({'status': 'SMS sent successfully.'})
    else:
        return jsonify({'error': 'Failed to send SMS. Check server logs and Twilio config.'}), 500


@app.route('/api/upload-and-analyze', methods=['POST'])
def upload_and_analyze_route():
    if 'video' not in request.files: return jsonify({'error': 'No video file part'}), 400
    file = request.files['video']
    if file.filename == '': return jsonify({'error': 'No selected file'}), 400
    temp_filename = f"{uuid.uuid4()}_{file.filename}"; temp_video_path = os.path.join(UPLOAD_FOLDER, temp_filename); file.save(temp_video_path)
    try:
        analysis_summary, qwen_prompt, qwen_frames = yolo_yamnet_analyzer.analyze_video_full(video_path_main=temp_video_path, yolo_model_path_override=YOLO_MODEL_PATH_CONFIG)
        def stream_response():
            with app.app_context():
                try: yolo_summary_json_string = jsonify(analysis_summary).get_data(as_text=True)
                except Exception: yolo_summary_json_string = '{"error": "Failed to serialize summary."}'
                yield f'{{"temp_filename": {json.dumps(temp_filename)}, "yolo_yamnet_summary": {yolo_summary_json_string}, "qwen_analysis_result": "'
                qwen_frame_data = [f['frame_image_base64'] for f in qwen_frames] if qwen_frames else []
                for chunk in call_qwen_api_stream(qwen_frame_data, qwen_prompt): yield chunk.replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
                yield '"}'
        return Response(stream_response(), mimetype='application/json')
    except Exception as e:
        if os.path.exists(temp_video_path): os.remove(temp_video_path)
        return jsonify({'error': f'Error during deep analysis: {str(e)}'}), 500

@app.route('/api/ask-question', methods=['POST'])
def ask_question_route():
    data = request.get_json(); question = data.get('question'); relative_video_path = data.get('video_path'); temp_filename = data.get('temp_filename')
    if not question: return jsonify({'error': 'A question is required.'}), 400
    if relative_video_path: absolute_video_path = os.path.join(VIDEO_BASE_PATH_ON_SERVER, relative_video_path)
    elif temp_filename: absolute_video_path = os.path.join(UPLOAD_FOLDER, temp_filename)
    else: return jsonify({'error': 'A video_path or temp_filename is required.'}), 400
    if not os.path.exists(absolute_video_path): return jsonify({'error': 'Video file not found on server.'}), 404
    try:
        num_frames = 30 if temp_filename else 12; frames_b64 = extract_evenly_spaced_frames_base64(absolute_video_path, num_frames)
        if not frames_b64: return jsonify({'error': 'Could not extract frames from video.'}), 500
        def stream_vqa_response():
            for chunk in call_qwen_api_stream(frames_b64, question): yield chunk
        return Response(stream_vqa_response(), mimetype='text/plain')
    except Exception as e: return jsonify({'error': f'Error during VQA: {str(e)}'}), 500

@app.route('/api/cleanup-upload', methods=['POST'])
def cleanup_upload_route():
    data = request.get_json(); temp_filename = data.get('temp_filename')
    if not temp_filename: return jsonify({'error': 'temp_filename is required.'}), 400
    video_path = os.path.join(UPLOAD_FOLDER, temp_filename)
    if os.path.exists(video_path):
        try: os.remove(video_path); print(f"[CLEANUP] Deleted temporary file: {temp_filename}"); return jsonify({'status': 'cleaned'}), 200
        except Exception as e: return jsonify({'error': f'Could not clean up file: {e}'}), 500
    return jsonify({'status': 'not_found'}), 200

if __name__ == '__main__':
    if not twilio_client:
        print("--- 🚨 WARNING 🚨 ---\nTwilio credentials are NOT fully set in .env file.\nSMS alerts for high threats will be disabled.\n--------------------")
    else:
        print(f"✅ Twilio SMS alerts configured to send from {TWILIO_PHONE_NUMBER}")
    if not os.path.exists(UPLOAD_FOLDER): os.makedirs(UPLOAD_FOLDER)
    if not QWEN_API_URL: print("--- 🚨 WARNING 🚨 ---\nQWEN_API_URL is NOT set.\n--------------------")
    else: print(f"✅ Qwen API URL configured to: {QWEN_API_URL}")
    if not os.path.exists(FACE_DATABASE_PATH): os.makedirs(FACE_DATABASE_PATH)
    if not os.path.exists(os.path.join(BASE_DIR, 'static', 'videos')): os.makedirs(os.path.join(BASE_DIR, 'static', 'videos'))
    print(f"✅ Face database ready at: {FACE_DATABASE_PATH}")
    if not os.path.exists(YOLO_MODEL_PATH_CONFIG): print(f"--- WARNING --- YOLO MODEL NOT FOUND at: {YOLO_MODEL_PATH_CONFIG}")
    else: print(f"✅ YOLO Model found at: {YOLO_MODEL_PATH_CONFIG}")
    app.run(host='0.0.0.0', port=9003, debug=False, use_reloader=False)