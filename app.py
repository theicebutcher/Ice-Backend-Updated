import os
import base64
from flask import Flask, request, jsonify, session, send_file
import openai
from dotenv import load_dotenv
import json
from flask_cors import CORS
import uuid
from PIL import Image
import io
import logging
import requests
import datetime
import time
import jwt
from google import genai
from google.genai import types
import pathlib
from werkzeug.utils import secure_filename
import tempfile
import shutil
import hashlib
from rapidfuzz import fuzz

# Configure logging
logging.basicConfig(level=logging.DEBUG)
Image.MAX_IMAGE_PIXELS = 999999999 

# ---------------- Image / Memory Optimization Constants ----------------
MAX_SOURCE_SIDE = int(os.getenv("MAX_SOURCE_SIDE", "2048"))
WORKING_THUMB_SIDE = int(os.getenv("WORKING_THUMB_SIDE", "1024"))
MAX_COMBINE_MEMORY_BYTES = int(os.getenv("MAX_COMBINE_MEMORY_BYTES", str(150 * 1024 * 1024)))
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "85"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(5 * 1024 * 1024)))
ENABLE_MEM_LOG = os.getenv("ENABLE_MEM_LOG", "0") == "1"

def optimize_saved_image(path: str):
    """Shrink and recompress an already-saved upload in-place to save RAM + disk."""
    try:
        if not os.path.exists(path):
            return
        with Image.open(path) as im:
            im = im.convert("RGB")
            if im.width > MAX_SOURCE_SIDE or im.height > MAX_SOURCE_SIDE:
                im.thumbnail((MAX_SOURCE_SIDE, MAX_SOURCE_SIDE), Image.LANCZOS)
            im.save(path, format='JPEG', quality=JPEG_QUALITY, optimize=True)
    except Exception as e:
        logging.warning(f"optimize_saved_image failed for {path}: {e}")

def log_memory(prefix: str):
    if not ENABLE_MEM_LOG:
        return
    try:
        import psutil, os as _os
        proc = psutil.Process(_os.getpid())
        rss = proc.memory_info().rss / (1024 * 1024)
        logging.debug(f"{prefix} RSS: {rss:.1f} MB")
    except Exception:
        pass

def save_generated_image(b64_data: str, upload_dir: str, base_name: str):
    target_format = os.getenv("OUTPUT_IMAGE_FORMAT", "png").lower()
    tmp_png_path = os.path.join(upload_dir, f"{base_name}.png")
    try:
        with open(tmp_png_path, 'wb') as f:
            f.write(base64.b64decode(b64_data))
    except Exception as e:
        logging.error(f"Failed writing raw generated image: {e}")
        raise
    if target_format in ("jpg", "jpeg"):
        try:
            with Image.open(tmp_png_path) as im:
                im = im.convert("RGB")
                jpeg_path = os.path.join(upload_dir, f"{base_name}.jpg")
                im.save(jpeg_path, format='JPEG', quality=JPEG_QUALITY, optimize=True)
            try:
                if os.path.getsize(jpeg_path) < os.path.getsize(tmp_png_path):
                    os.remove(tmp_png_path)
                    return jpeg_path
            except OSError:
                return jpeg_path
            return jpeg_path
        except Exception as e:
            logging.warning(f"Transcode to JPEG failed, keeping PNG: {e}")
            return tmp_png_path
    return tmp_png_path

app = Flask(__name__)

# Configure CORS
CORS(app, 
     resources={r"/api/*": {
         "origins": [
             "https://ice-frontend-beta.vercel.app",
             "https://ice-frontend-hyj0t73fw-alihamzasultans-projects.vercel.app",
             "https://*.vercel.app",
             "http://localhost:5173",
             "http://localhost:3000"
         ],
         "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
         "allow_headers": ["Content-Type", "Authorization"],
         "supports_credentials": True,
         "max_age": 3600
     }})

app.secret_key = os.getenv("FLASK_SECRET_KEY", "default-secret-key")
app.config["UPLOAD_FOLDER"] = "static/uploads"
app.config["GENERATED_FOLDER"] = "static/generated"
os.makedirs(app.config["GENERATED_FOLDER"], exist_ok=True)

# Load APIs
load_dotenv()
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Cloudinary
import cloudinary
import cloudinary.uploader
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

# Initialize Supabase
from supabase import create_client, Client
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

# Initialize Supabase Admin
# Initialize Supabase Admin
service_role_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") 
if not service_role_key: 
    logging.warning("SUPABASE_SERVICE_ROLE_KEY is missing. Admin user fetching will be disabled.")
    # Do NOT fallback to SUPABASE_KEY as it cannot perform admin actions

supabase_admin: Client = None
if service_role_key:
    try:
        supabase_admin = create_client(supabase_url, service_role_key)
        logging.info("Supabase Admin client initialized")
    except Exception as e:
        logging.error(f"Failed to initialize Supabase Admin client: {e}")

# Gemini Client
client2 = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
PRO_MODEL_ID = "gemini-3-pro-image-preview"

# ---------------- Dictionaries & Constants ----------------

SCULPTURE_BASES = {
    "champagne": os.path.join("bases", "champagne_base.png"),
    "crystal base": os.path.join("bases", "crystal_base.png"),
    "frame base": os.path.join("bases", "frame_base.png"),
    "golf base": os.path.join("bases", "golf_base.png"),
    "heart base": os.path.join("bases", "heartBase.png"),
    "plynth base w logo": os.path.join("bases", "Plynth base w logo.png"),
    "rings base": os.path.join("bases", "rings base.png"),
    "star base": os.path.join("bases", "Star base.png"),
    "swirl base": os.path.join("bases", "swirl base.png"),
    "tilt plynth base": os.path.join("bases", "tilt plynth base.png"),
    "plynth base": os.path.join("bases", "plynth_base.png"),
    "ring base": os.path.join("bases", "ring_base.png"),
    "tee base": os.path.join("bases", "tee_base.png"),
    "waves base": os.path.join("bases", "waves_base.png"),
    "double ludge": os.path.join("sculptures", "double_ludge.png"),
    "martini": os.path.join("sculptures", "martini.png"),
    "tube": os.path.join("sculptures", "tube.png"),
    "alligator head": os.path.join("sculptures", "alligator_head.png"),
    "alligator": os.path.join("sculptures", "alligator.png"),
    "anchor": os.path.join("sculptures", "anchor.png"),
    "crab claw": os.path.join("sculptures", "crab claw.png"),
    "dolphin": os.path.join("sculptures", "dolphin.png"),
    "dragon head": os.path.join("sculptures", "dragon_head.png"),
    "griffen": os.path.join("sculptures", "griffen.png"),
    "guitar": os.path.join("sculptures", "guitar.png"),
    "heel": os.path.join("sculptures", "heel.png"),
    "horse head": os.path.join("sculptures", "horse_head.png"),
    "indian head": os.path.join("sculptures", "indian head.png"),
    "leopard": os.path.join("sculptures", "leopard.png"),
    "lion": os.path.join("sculptures", "lion.png"),
    "lobster": os.path.join("sculptures", "lobster.png"),
    "mahi mahi": os.path.join("sculptures", "mahi mahi.png"),
    "mask": os.path.join("sculptures", "mask.png"),
    "mermaid": os.path.join("sculptures", "mermaid.png"),
    "palm_trees": os.path.join("sculptures", "palm_trees.png"),
    "panther": os.path.join("sculptures", "panther.png"),
    "penguin": os.path.join("sculptures", "penguin.png"),
    "selfish": os.path.join("sculptures", "selfish.png"),
    "shark": os.path.join("sculptures", "shark.png"),
    "shrimp": os.path.join("sculptures", "shrimp.png"),
    "turkey": os.path.join("sculptures", "turkey.png"),
    "turle": os.path.join("sculptures", "turle.png"),
    "turtle cartoon": os.path.join("sculptures", "turtle_cartoon.png"),
    "unicorn": os.path.join("sculptures", "unicorn.png"),
    "vase": os.path.join("sculptures", "vase.png"),
    "whale": os.path.join("sculptures", "whale.png"),
    "women butt": os.path.join("sculptures", "women_butt.png"),
    "women torso": os.path.join("sculptures", "women_torso.png"),
    "interlocking rings": os.path.join("wedding_Showpieces", "interlocking_rings_showpiece_wedding.png"),
    "wedding frame": os.path.join("wedding_Showpieces", "picture_frame_wedding.png"),
    "banana single luge": os.path.join("Toppers", "banana single luge.jpg"),
    "ice bar mini single luge": os.path.join("Toppers", "ice bar mini single luge.jpg"),
    "ice bowl": os.path.join("Toppers", "ice bowl.png"),
    "crown logo as topper": os.path.join("Toppers", "crown logo as topper.jpg"),
    "6ft ice bar": os.path.join("Ice bars", "6ft ice bar.jpg"),
    "8ft ice bar":os.path.join("Ice bars", "8ft ice bar.jpg"),   
    "12ft ice bar": os.path.join("Ice bars", "12ft ice bar.jpg"),   
}

LUDGE_TYPES = {
    "martini": os.path.join("sculptures", "martini.png"),
    "tube": os.path.join("sculptures", "tube.png"),
    "double": os.path.join("sculptures", "double_ludge.png")
}

ICE_CUBE_PROMPTS = {
    "Snofilled": """
    "task": "add the logo image into the center of the icecube",
    "instructions": 
        "effect": "Create a carved snow-filled appearance inside the ice sculpture, the image should not be colored and it should be engraved into the icecube with some depth",
        "ice":"ice should be realistic WHITE ice, crystal clear, transparent, and completely free of any cloudiness, bubbles, or impurities. The ice must appear as pure white/clear ice like real ice sculptures. DO NOT make the ice blue - ice should be white or transparent like natural ice",
        "Strict": "the logo should be engraved into the ice few centimeters with some depth. The ice cube shape MUST be an EXACT CUBE. Do NOT distort the cube shape.",
        "Extra":"remove any background of the image before adding it to the icecube. PRESERVE THE LOGO FONT AND GEOMETRY EXACTLY."
    """,
    "Colored": """
    "task": "add the logo image into the center of the icecube ", 
    "instructions": 
        "effect": "it should look like the ice is colored and not etched",
        "Strict": "the image should be engraved into the ice few centimeters with some depth",
        "Extra":"remove any background of the image before adding it to the icecube",
        "ice":"ice should be realistic WHITE ice, crystal clear, transparent, and completely free of any cloudiness, bubbles, or impurities. The ice must appear as pure white/clear ice like real ice sculptures. DO NOT make the ice blue - ice should be white or transparent like natural ice"
    """,
    "Paper": """
    "task": "add the image inside the icecube, ",
    "instructions": 
        "effect": "it should look like a colored printed paper is frozen into the icecube, the Logo should be colored with some white outline and transparent background and should be in center of the cube",
        "Strict": "the image should be placed into the ice few centimeters in some depth",
        "Extra":"remove any background of the image before adding it to the icecube",
        "ice":"ice should be realistic WHITE ice, crystal clear, transparent, and completely free of any cloudiness, bubbles, or impurities. The ice must appear as pure white/clear ice like real ice sculptures. DO NOT make the ice blue - ice should be white or transparent like natural ice, increase the size of the cube if the logo doesnot fit"
    """,
    "Snofilled+paper": """
    "task": "add the image into the center of the icecube ",
    "instructions": 
        "effect": "it should look like a colored printed paper is frozen into the icecube, the paper should be colored and should be in center of the cube, and the ice should be etched a little bit on the outlines of the image logo",
        "Strict": "the logo should be engraved into the ice few centimeters with some depth",
        "Extra":"remove any background of the image before adding it to the icecube",
        "ice":"ice should be realistic WHITE ice, crystal clear, transparent, and completely free of any cloudiness, bubbles, or impurities. The ice must appear as pure white/clear ice like real ice sculptures. DO NOT make the ice blue - ice should be white or transparent like natural ice"
    """
}

# ---------------- Core Helper Functions ----------------

def get_current_user_id():
    """
    Extract user ID from Supabase auth token in request headers.
    Returns user_id if authenticated, None otherwise.
    """
    try:
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            logging.warning("No Authorization header found")
            return None
        
        token = auth_header.replace('Bearer ', '')
        
        # Use Supabase to verify the JWT token
        try:
            user = supabase.auth.get_user(token)
            if user and user.user:
                logging.info(f"Authenticated user: {user.user.id}")
                return user.user.id
        except Exception as e:
            logging.warning(f"Token verification failed: {e}")
            # Fallback: decode JWT without verification (less secure but works for some flows)
            decoded = jwt.decode(token, options={"verify_signature": False})
            user_id = decoded.get('sub')
            if user_id:
                logging.info(f"Extracted user_id from token: {user_id}")
                return user_id
        
        return None
    except Exception as e:
        logging.error(f"Error extracting user ID: {e}")
        return None

def upload_and_save_generated_image(file_path, prompt, template_type, original_filename, template_name=None, user_id=None):
    """
    Uploads a generated image to Cloudinary and saves its URL and metadata to Supabase.
    """
    try:
        # Upload to Cloudinary
        upload_result = cloudinary.uploader.upload(
            file_path,
            folder="generated_images",
            public_id=os.path.splitext(original_filename)[0]
        )
        image_url = upload_result.get("secure_url")
        
        # Insert into Supabase
        data = {
            "image_url": image_url,
            "prompt": prompt,
            "template_type": template_type,
            "original_filename": original_filename,
            "template_name": template_name,
            "user_id": user_id
        }
        
        supabase.table("generated_images").insert(data).execute()
        logging.info(f"Successfully saved generated image to Supabase: {image_url}")
        
        # Clean up local file after successful upload
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            logging.warning(f"Failed to delete local file {file_path}: {e}")
            
        return image_url
    except Exception as e:
        logging.error(f"Error uploading/saving generated image: {e}")
        return None

def combine_images(image_paths, output_path, max_size=1024):
    """
    Combines multiple images into one with a white background.
    """
    unique_images = []
    seen_hashes = set()
    est_total_bytes = 0
    for path in image_paths:
        try:
            with Image.open(path) as img:
                img = img.convert("RGB")
                if img.width > MAX_SOURCE_SIDE or img.height > MAX_SOURCE_SIDE:
                    img.thumbnail((MAX_SOURCE_SIDE, MAX_SOURCE_SIDE), Image.LANCZOS)
                img.thumbnail((max_size, max_size), Image.LANCZOS)
                img_bytes = img.tobytes()
                img_hash = f"{img.width}x{img.height}-" + hashlib.md5(img_bytes).hexdigest()
                if img_hash in seen_hashes:
                    continue
                seen_hashes.add(img_hash)
                est_total_bytes += len(img_bytes)
                if est_total_bytes > MAX_COMBINE_MEMORY_BYTES:
                    logging.warning("combine_images aborted: memory estimate exceeded limit")
                    break
                unique_images.append(img.copy())
        except Exception as e:
            logging.error(f"Error processing {path}: {e}")
            continue
    if not unique_images:
        raise ValueError("No valid images to combine")
    max_height = max(im.height for im in unique_images)
    normalized = []
    for im in unique_images:
        if im.height < max_height:
            ratio = max_height / im.height
            new_w = int(im.width * ratio)
            im = im.resize((new_w, max_height), Image.LANCZOS)
        normalized.append(im)
    total_width = sum(im.width for im in normalized)
    new_im = Image.new('RGB', (total_width, max_height), 'white')
    x_off = 0
    for im in normalized:
        new_im.paste(im, (x_off, 0))
        x_off += im.width
    new_im.save(output_path, format='JPEG', quality=JPEG_QUALITY, optimize=True)
    logging.info(f"Combined image saved at {output_path}")

def detect_ludge_type(input_text):
    input_text = input_text.lower()
    for ludge in LUDGE_TYPES:
        if ludge in input_text:
            return LUDGE_TYPES[ludge]
    return None

def detect_sculpture_bases(input_text, threshold=80):
    """Detects which sculpture bases are mentioned in the input text."""
    input_text = input_text.lower()
    detected_bases = []
    
    input_without_base = input_text.replace("base", "").strip()
    if not input_without_base:
        input_without_base = input_text
    
    for keyword, image_path in SCULPTURE_BASES.items():
        main_keyword = keyword.replace("base", "").strip()
        if main_keyword and main_keyword in input_without_base:
            detected_bases.append(image_path)
        elif keyword in input_text:
            detected_bases.append(image_path)
        else:
            words = input_without_base.split()
            if any(fuzz.ratio(word, main_keyword) >= threshold for word in words):
                detected_bases.append(image_path)
    return detected_bases

def classify_prompt_type(prompt):
    classification_messages = [
        {
            "role": "system",
            "content": "You are a classification assistant. Based on the user's input, decide if they are asking for a sculpture to be generated, edit an image, or just get a text response. Reply with only one word: 'generate', 'edit', or 'text'."
        },
        {"role": "user", "content": prompt}
    ]
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=classification_messages
    )
    return response.choices[0].message.content.strip().lower()

# ---------------- Routes ----------------

@app.route('/api/admin/get_history', methods=['POST'])
def admin_get_history():
    try:
        # 1. Verify Authentication & Admin Status
        user_id = get_current_user_id()
        if not user_id:
            return jsonify({'error': 'Unauthorized'}), 401
        
        # 2. Get Pagination Params
        data = request.get_json() or {}
        page = data.get('page', 0)
        limit = data.get('limit', 50)
        start = page * limit
        end = start + limit - 1

        # 3. Fetch Images
        response = supabase.table('generated_images').select('*').order('created_at', desc=True).range(start, end).execute()
        images = response.data

        # 4. Fetch User Emails
        user_ids = list(set([img['user_id'] for img in images if img.get('user_id')]))
        user_map = {}
        if user_ids and supabase_admin:
            print(f"DEBUG: Fetching details for {len(user_ids)} users via Supabase Admin")
            for uid in user_ids:
                try:
                    # Use supabase_admin to fetch user details
                    print(f"DEBUG: Fetching user {uid}...")
                    u_res = supabase_admin.auth.admin.get_user_by_id(uid)
                    if u_res and u_res.user:
                        print(f"DEBUG: Found user {uid}. Email: {u_res.user.email}, Meta: {u_res.user.user_metadata}")
                        user_map[uid] = {
                            'email': u_res.user.email,
                            'display_name': u_res.user.user_metadata.get('display_name') if u_res.user.user_metadata else None,
                            'avatar_url': u_res.user.user_metadata.get('avatar_url') if u_res.user.user_metadata else None
                        }
                    else:
                         print(f"DEBUG: User {uid} not found or no user object returned.")
                except Exception as e:
                    logging.warning(f"Failed to fetch user {uid}: {e}")
                    print(f"DEBUG: Exception fetching user {uid}: {e}")
                    user_map[uid] = {'email': None, 'display_name': None, 'avatar_url': None}
        elif not supabase_admin:
            print("DEBUG: Supabase Admin client is MISSING or None")
            logging.warning("Supabase Admin client not available. Cannot fetch emails.")

        # 5. Merge Data
        for img in images:
            uid = img.get('user_id')
            if uid and uid in user_map:
                img['email'] = user_map[uid]['email']
                img['display_name'] = user_map[uid]['display_name']
                img['avatar_url'] = user_map[uid]['avatar_url']
        
        return jsonify({'data': images}), 200

    except Exception as e:
        logging.error(f"Error in admin_get_history: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/submit_feedback', methods=['POST'])
def submit_feedback():
    try:
        data = request.get_json()
        image_url = data.get('image_url')
        rating = data.get('rating')
        comment = data.get('comment')
        if not image_url or not rating:
            return jsonify({'error': 'Missing image_url or rating'}), 400
        base64_image = ""
        
        if image_url.startswith('http'):
            try:
                img_response = requests.get(image_url)
                if img_response.status_code == 200:
                    base64_image = base64.b64encode(img_response.content).decode('utf-8')
                else:
                    return jsonify({'error': f'Failed to download image from URL: {image_url}'}), 400
            except Exception as e:
                return jsonify({'error': f'Error downloading image: {str(e)}'}), 500
        else:
            base_dir = os.path.abspath(os.path.dirname(__file__))
            clean_path = image_url.replace('/static/', '').replace('static/', '')
            image_path = os.path.join(base_dir, 'static', clean_path)
            if not os.path.exists(image_path):
                return jsonify({'error': f'Image file not found at {image_path}'}), 404
            with open(image_path, 'rb') as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        
        feedback_data = {
            'imageData': base64_image,
            'rating': rating,
            'comment': comment or 'No comment provided'
        }
        script_url = 'https://script.google.com/macros/s/AKfycbzKOxVD9ju-bewiQuCZS8hHByJ2JfU0mhhDbQFxWYIaQemRPgeJLtrvbOC8G-yf3vmg/exec'
        response = requests.post(script_url, json=feedback_data, timeout=30)
        if response.status_code != 200:
            return jsonify({'error': f'Failed to save feedback, status: {response.status_code}'}), 500
        return jsonify({'message': 'Feedback submitted successfully'}), 200
    except Exception as e:
        logging.error(f"Error in submit_feedback: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/delete_history_item', methods=['POST'])
def delete_history_item():
    try:
        data = request.get_json()
        item_id = data.get('id')
        if not item_id:
            return jsonify({'error': 'Missing item id'}), 400
        response = supabase.table('generated_images').delete().eq('id', item_id).execute()
        return jsonify({'status': 'success', 'message': 'Item deleted successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/toggle_favourite', methods=['POST'])
def toggle_favourite():
    try:
        data = request.get_json()
        item_id = data.get('id')
        is_favourite = data.get('is_favourite')
        if not item_id:
            return jsonify({'error': 'Missing item id'}), 400
        response = supabase.table('generated_images').update({'is_favourite': is_favourite}).eq('id', item_id).execute()
        return jsonify({'status': 'success', 'message': 'Favourite status updated'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/get_user_history', methods=['GET'])
def get_user_history():
    try:
        user_id = get_current_user_id()
        if not user_id:
            return jsonify({'error': 'Unauthorized'}), 401
        response = supabase.table('generated_images').select('*').eq('user_id', user_id).order('created_at', desc=True).execute()
        return jsonify({'data': response.data}), 200
    except Exception as e:
        logging.error(f"Error fetching user history: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/get_user_favourites', methods=['GET'])
def get_user_favourites():
    try:
        user_id = get_current_user_id()
        if not user_id:
            return jsonify({'error': 'Unauthorized'}), 401
        response = supabase.table('generated_images').select('*').eq('user_id', user_id).eq('is_favourite', True).order('created_at', desc=True).execute()
        return jsonify({'data': response.data}), 200
    except Exception as e:
        logging.error(f"Error fetching user favourites: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/template_selected', methods=['POST'])
def handle_template_selection():
    data = request.get_json()
    template_type = data.get('template')
    template_name = data.get('templateName', '')
    
    session['selected_template_name'] = template_name
    if template_type in ICE_CUBE_PROMPTS:
        session['selected_ice_cube'] = template_type
        print(f"Ice cube selected: {template_type}")
    elif "ice bar" not in template_type.lower() and "ice cube" not in template_type.lower():
        session['template_selected_message'] = "Add a silver plastic rectangular ambient diffused blue lighting at the bottom of the sculpture, must be very dimm light and hidden"
    return jsonify({"status": "success", "message": "Template selection received"})

@app.route('/api/extract_logo', methods=['POST'])
def extract_logo():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        file = request.files['file']
        file.stream.seek(0, os.SEEK_END)
        size = file.stream.tell()
        file.stream.seek(0)
        if size > MAX_UPLOAD_BYTES:
            return jsonify({"error": f"File too large. Max {MAX_UPLOAD_BYTES//1024//1024}MB"}), 400
        
        with tempfile.TemporaryDirectory() as temp_dir:
            filename = secure_filename(file.filename)
            upload_path = os.path.join(temp_dir, filename)
            file.save(upload_path)
            optimize_saved_image(upload_path)

            with Image.open(upload_path) as img:
                img = img.convert('RGB')
                if img.width > WORKING_THUMB_SIDE or img.height > WORKING_THUMB_SIDE:
                    img.thumbnail((WORKING_THUMB_SIDE, WORKING_THUMB_SIDE), Image.LANCZOS)
                model_image = img.copy()

            prompt = "Extract the logo from this image and display it on a pure white background, tightly cropped."
            response = client2.models.generate_content(
                model=PRO_MODEL_ID,
                contents=[prompt, model_image],
                config=types.GenerateContentConfig(response_modalities=['Text', 'Image'])
            )
            for part in response.candidates[0].content.parts:
                if part.inline_data is not None:
                    data = part.inline_data.data
                    logo_path = os.path.join(temp_dir, 'extracted_logo.png')
                    pathlib.Path(logo_path).write_bytes(data)
                    optimize_saved_image(logo_path)
                    return send_file(logo_path, mimetype='image/png')
            return jsonify({"error": "No image returned from model"}), 500
    except Exception as e:
        print("Error extracting logo:", str(e))
        return jsonify({"error": str(e)}), 500

@app.route('/api/log_button_press', methods=['POST'])
def log_button_press():
    data = request.get_json()
    image_url = data.get('image_url')
    timestamp = data.get('timestamp')
    print(f"Expand button pressed for image: {image_url} at {timestamp}")
    return jsonify({'status': 'success'})

@app.route("/api/expand_chatbot", methods=["POST"])
def expand_chatbot():
    user_id = get_current_user_id()
    user_input = request.form.get("user_input", "").strip()
    uploaded_files = request.files.getlist("images")
    template_name = request.form.get("template_name") or session.get('selected_template_name')

    try:
        if not uploaded_files:
            return jsonify({"response": "No image provided for expansion"}), 400
        
        with tempfile.TemporaryDirectory() as temp_dir:
            uploaded_file = uploaded_files[0]
            uploaded_file.stream.seek(0, os.SEEK_END)
            size = uploaded_file.stream.tell(); uploaded_file.stream.seek(0)
            if size > MAX_UPLOAD_BYTES:
                return jsonify({"response": f"File too large. Max {MAX_UPLOAD_BYTES//1024//1024}MB"}), 400
            
            combined_id = uuid.uuid4().hex[:8]
            combined_path = os.path.join(temp_dir, f"combined_{combined_id}.jpg")
            uploaded_file.save(combined_path)
            optimize_saved_image(combined_path)

            with Image.open(combined_path) as img:
                img = img.convert('RGB')
                response = client2.models.generate_content(
                    model=PRO_MODEL_ID,
                    contents=[user_input, img],
                    config=types.GenerateContentConfig(response_modalities=['Image'])
                )

            output_id = uuid.uuid4().hex[:8]
            output_filename = f"sculpture_{output_id}.png"
            output_path = os.path.join(temp_dir, output_filename)
            
            for part in response.candidates[0].content.parts:
                if part.inline_data is not None:
                    pathlib.Path(output_path).write_bytes(part.inline_data.data)
                    optimize_saved_image(output_path)
                    break
            
            final_image_url = upload_and_save_generated_image(
                output_path, "Expand Image", "expand", output_filename,
                template_name=template_name, user_id=user_id
            )
            
            if not final_image_url:
                 return jsonify({"response": "Error saving generated image"}), 500
            return jsonify({"image_url": final_image_url})
    except Exception as e:
        logging.exception("expand_chatbot failed")
        return jsonify({"response": f"Error expanding image: {str(e)}"}), 500
    
@app.route("/api/chatbot", methods=["POST"])
def chatbot():
    user_id = get_current_user_id()
    
    if "conversation" not in session or not isinstance(session["conversation"], list):
        session["conversation"] = []
    image_generation_prompt = None
    if len(session["conversation"]) > 5:
        session["conversation"] = session["conversation"][-5:]

    user_input = request.form.get("user_input", "").strip()
    uploaded_files = request.files.getlist("images")
    user_aspect_ratio = request.form.get("aspect_ratio", "9:16")
    template_name = request.form.get("template_name") or session.get('selected_template_name')

    selected_ice_cube = session.get('selected_ice_cube')
    
    if selected_ice_cube and uploaded_files:
        ice_prompt = ICE_CUBE_PROMPTS.get(selected_ice_cube, "")
        image_generation_prompt = f"{ice_prompt}\nUSER INPUT:\n{user_input}"
        session.pop('selected_ice_cube', None)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            uploaded_paths = []
            cloudinary_urls = []
            
            for file in uploaded_files:
                temp_path = os.path.join(temp_dir, secure_filename(file.filename))
                file.stream.seek(0, os.SEEK_END)
                size = file.stream.tell(); file.stream.seek(0)
                if size > MAX_UPLOAD_BYTES:
                    return jsonify({"response": f"One of the files is too large (>{MAX_UPLOAD_BYTES//1024//1024}MB)"})
                file.save(temp_path)
                optimize_saved_image(temp_path)
                uploaded_paths.append(temp_path)
                
                try:
                    upload_result = cloudinary.uploader.upload(
                        temp_path, folder="input_images", public_id=f"input_{uuid.uuid4().hex[:8]}"
                    )
                    cloudinary_urls.append(upload_result.get("secure_url"))
                except Exception as e:
                    logging.error(f"Failed to upload input to Cloudinary: {e}")

            combined_path = os.path.join(temp_dir, f"combined_{uuid.uuid4().hex[:8]}.jpg")
            if uploaded_paths:
                try:
                    combine_images(uploaded_paths, combined_path)
                except ValueError as e:
                    return jsonify({"response": str(e)})
            else:
                return jsonify({"response": "No valid images provided"})
            
            try:
                with Image.open(combined_path) as img:
                    img = img.convert('RGB')
                    response = client2.models.generate_content(
                        model=PRO_MODEL_ID,
                        contents=[image_generation_prompt, img],
                        config=types.GenerateContentConfig(response_modalities=['Image'])
                    )
                
                output_id = uuid.uuid4().hex[:8]
                output_filename = f"sculpture_{output_id}.png"
                output_path = os.path.join(temp_dir, output_filename)
                
                for part in response.candidates[0].content.parts:
                    if part.inline_data is not None:
                        pathlib.Path(output_path).write_bytes(part.inline_data.data)
                        optimize_saved_image(output_path)
                        break
                
                final_image_url = upload_and_save_generated_image(
                    output_path, image_generation_prompt, "ice_cube", output_filename,
                    template_name=template_name, user_id=user_id
                )
                if not final_image_url:
                     return jsonify({"response": "Error saving generated image"}), 500

                session["conversation"].append({"role": "user", "content": user_input, "images": cloudinary_urls})
                session["conversation"].append({
                    "role": "assistant", "content": "Here is your ice sculpture:", "image": final_image_url
                })
                session.modified = True
                return jsonify({"image_url": final_image_url})
            except Exception as e:
                return jsonify({"response": f"Error generating ice cube image: {str(e)}"})

    user_input_lower = user_input.lower()
    detected_ludge = detect_ludge_type(user_input_lower)

    if "ludge" in user_input_lower and not detected_ludge:
        response = "Can you please specify which ludge? We have martini ludge, tube ludge, and double ludge."
        session["conversation"].append({"role": "user", "content": user_input})
        session["conversation"].append({"role": "assistant", "content": response})
        session.modified = True
        return jsonify({"response": response})

    base_images = detect_sculpture_bases(user_input_lower)
    if detected_ludge:
        base_images.append(detected_ludge)
        
    if uploaded_files or base_images:
        temp_dir = tempfile.mkdtemp()
        uploaded_paths = []
        
        if uploaded_files:
            for file in uploaded_files:
                temp_path = os.path.join(temp_dir, secure_filename(file.filename))
                file.stream.seek(0, os.SEEK_END)
                size = file.stream.tell(); file.stream.seek(0)
                if size > MAX_UPLOAD_BYTES:
                    shutil.rmtree(temp_dir)
                    return jsonify({"response": f"One of the files is too large"})
                file.save(temp_path)
                optimize_saved_image(temp_path)
                uploaded_paths.append(temp_path)
        
        base_image_paths = [os.path.join("static", img) for img in base_images] if base_images else []
        all_image_paths = uploaded_paths + base_image_paths
        
        if not all_image_paths:
             shutil.rmtree(temp_dir)
             return jsonify({"response": "No valid images provided"})
        
        is_ice_bar = "ice bar" in user_input_lower or any("ice bars" in img.lower() for img in base_images)
        
        image_generation_dict = {
            "user_input": user_input,
            "Sculpture_instructions": {
                "sculpture_preservation": {
                    "shape": "Maintain EXACT shape, proportions, and details from the reference image",
                    "alterations": "Do NOT alter, add, or remove any elements of the sculpture",
                    "Extra_ice": "Do NOT ADD EXTRA ICE TO THE SCULPTURE, ONLY THE ORIGINAL IMAGE SHOULD BE USED",
                    "contours": "Preserve all original contours and features precisely",
                    "size": "Sculpture should be large, around 6 to 7 feet tall or wide accordingly",
                    "color_coding": "CRITICAL: Blue color in the input reference image is ONLY A TEMPLATE INDICATOR showing where ice should be. The OUTPUT sculpture MUST BE COMPLETELY WHITE/CLEAR TRANSPARENT ICE, NOT BLUE. Light blue in reference means recess in the ice. Any other color in reference means it is made of paper and not ice. TRANSFORM ALL BLUE PARTS INTO REALISTIC WHITE/CLEAR TRANSPARENT ICE IN THE FINAL IMAGE. The final result should NEVER show blue ice - only clear/white transparent ice.",
                    "CRITICAL_NO_NEW_SCULPTURES": "DO NOT create new sculptures like deer, bear, animals, or any other sculptures that are not in the reference image. ONLY render what is shown in the uploaded reference images. If toppers are added, place them ON TOP of the existing sculpture - do not replace or add new main sculptures"
                },
                "material_properties": {
                    "rendering": "Render as completely TRANSPARENT and CLEAR ice like real ice sculptures. Ice should be see-through glass-like with no color, no white layers, no frosting, and ABSOLUTELY NO BLUE COLOR",
                    "lighting": "Include realistic light refraction through transparent ice. Light should pass through the ice, creating natural highlights and reflections",
                    "surface": "Surface should appear smooth, polished, and completely transparent",
                    "ice_clarity": "Ice must be 100% TRANSPARENT and CLEAR throughout. NO white layers, NO frosted sections, NO cloudiness, NO bubbles. The ice should be completely see-through like glass or water. MOST IMPORTANT: NO BLUE COLOR - ice must be clear/white transparent only",
                    "ice_color": "CRITICAL: Ice must be COMPLETELY TRANSPARENT and CLEAR like glass. DO NOT add white layers, frosted sections, or any color. DO NOT KEEP THE BLUE COLOR FROM THE TEMPLATE. The ice should be see-through so you can see through it. Only logos/text should have color, the ice itself must be crystal clear transparent WITHOUT ANY BLUE",
                    "no_white_layers": "DO NOT create white frosted layers or white sections in the ice. The entire ice sculpture must be transparent and clear like glass",
                    "BLUE_IS_TEMPLATE_ONLY": "The blue color you see in the input image is ONLY a template/reference color. DO NOT render the final ice as blue. Convert all blue areas to clear/white transparent ice in the output."
                },
                "background_environment": {
                    "placement": "Place the sculpture on a wooden table",
                    "setting": "Environment should be realistic, preferably a country club",
                    "camera_angle": "CRITICAL: Take the photo from DIRECTLY IN FRONT of the sculpture (front-facing view). The camera must be centered and positioned straight-on facing the front of the sculpture. DO NOT take the photo from the side, at an angle, or from a corner. The view must be head-on, frontal, and centered.",
                    "framing": "WIDE ANGLE VIEW: The sculpture should appear in the CENTER of the image with SIGNIFICANT SPACE around it. DO NOT zoom in or fill the entire frame. Show the full sculpture with empty space on all sides - approximately 30-40% of the image should be background/environment around the sculpture. Think of it like taking a photo from several feet away to capture the whole scene, not a close-up.",
                    "composition": "The sculpture should take up roughly 50-60% of the frame height/width, leaving visible gaps and breathing room around it. Show the table it sits on and the room behind it."
                },
                "prohibited_modifications": [
                    "DO NOT ADD ANYTHING THAT IS NOT REQUESTED BY USER",
                    "NO changes to the sculpture structure",
                    "NO additional decorative elements",
                    "NO human figures or living creatures",
                    "NO elements detached from the sculpture even if requested",
                    "NO small details",
                    "NO foggy ice",
                    "NO cloudy ice - ice must be crystal clear and transparent",
                    "NO bubbles or impurities in the ice",
                    "NO extra ice base for the sculpture",
                    "NO changes in the sculpture itself allowed",
                    "DO NOT change the sculpture design",
                    "NO extra ice pieces on the sculpture",
                    "DO NOT CREATE NEW SCULPTURES - only render what is in the reference images",
                    "DO NOT ADD deer, bear, animals, or any sculptures not shown in reference",
                    "DO NOT REPLACE the main sculpture - keep the exact sculpture from the uploaded images",
                    "Place the sculpture directly on the table without extra ice base",
                    "DO NOT add any company logos (e.g., 'ice butcher, purveyors of perfect ice')",
                    "DO NOT add any text, labels, or words to the sculpture unless explicitly requested by user",
                    "DO NOT add any logos, brand names, or company names unless explicitly provided by user",
                    "DO NOT interpret visual elements as text or add text based on visual patterns",
                    "DO NOT add placeholder text, sample text, or any written content",
                    "ONLY add text or logos if the user explicitly uploads them or requests them in their input",
                    "NO WHITE LAYERS in the ice - ice must be completely transparent and clear like glass",
                    "NO FROSTED SECTIONS - the entire ice sculpture must be see-through transparent",
                    "NO OPAQUE or MILKY areas - ice should be crystal clear throughout",
                    "NO WHITE BACKGROUNDS around logos or text - only the logo/text itself should be visible, embedded in clear ice",
                    "NO SIDE ANGLES - always photograph from directly in front (front-facing view)",
                    "NO DIAGONAL or CORNER perspectives - camera must be centered and head-on",
                    "NO ANGLED SHOTS - always shoot straight-on from the front",
                    "NO ZOOMED IN CLOSE-UPS - use wide angle view with space around sculpture",
                    "NO FILLING THE ENTIRE FRAME - sculpture should be centered with gaps around it",
                    "MUST SHOW CONTEXT - include table, background, and environment around the sculpture",
                    "NO BLUE ICE - the blue color in reference images is a template only, render all ice as clear/white transparent",
                    "DO NOT keep the blue color from template images - convert to clear transparent ice",
                    "BLUE = TEMPLATE COLOR ONLY, not the final ice color"
                ],
                "image_quality": "Always create an HD high-resolution image captured by a high-resolution camera",
                "photography_rules": {
                    "perspective": "MANDATORY: Front-facing, head-on view ONLY. NO side angles, NO diagonal views, NO corner perspectives",
                    "camera_position": "Position the camera directly in front of the sculpture at eye level, centered perfectly",
                    "framing": "The sculpture should face the camera directly, showing the front face of the sculpture clearly",
                    "zoom_level": "WIDE ANGLE SHOT: Camera should be positioned FAR ENOUGH BACK to show the sculpture in the center with plenty of space around it. DO NOT zoom in close. DO NOT fill the frame completely. Leave 30-40% empty space around the sculpture.",
                    "scale": "The sculpture should appear as if photographed from 6-8 feet away, showing the full context and environment, not a tight close-up"
                },
                "sculpture_image_rules": {
                    "stickers": {
                        "condition": "If any detail in the sculpture is other than blue",
                        "appearance": "It should look like a colored paper sticker pasted on the ice, not made of ice. NO white background or frosted layer around the sticker"
                    },
                    "transparency": "The ice must remain completely transparent and clear throughout. NO white frosted layers, NO opaque sections, NO milky areas. Only the embedded logos/text should have color, everything else is clear transparent ice"
                },
                "modular_components": {
                    "topper": {
                        "condition": "ONLY if the user explicitly requests a topper in their text input",
                        "instruction": "Place the topper ON TOP of the EXISTING sculpture from the reference image. Do NOT create new sculptures like deer, bear, or animals. Just add the small topper piece on top of what already exists. Do NOT add the text 'TOPPER'",
                        "restriction": "Do NOT modify the base sculpture in any way. Do NOT replace the sculpture with new animals or designs. Do NOT add toppers unless explicitly requested by user."
                    },
                    "topper_with_logo": {
                        "condition": "ONLY if the user explicitly requests a topper with logo in their text input",
                        "instruction": "Place it on top of the sculpture with a centered placeholder logo, but do NOT add a logo unless provided",
                        "restriction": "Do NOT modify the base sculpture in any way. Do NOT add toppers or logos unless explicitly requested by user."
                    },
                    "base": {
                        "condition": "ONLY if the user explicitly requests a base in their text input",
                        "instruction": "Place it directly at the bottom of the sculpture, do NOT add the text 'BASE'",
                        "restriction": "Do NOT modify the base sculpture in any way. Do NOT add bases unless explicitly requested by user."
                    }
                },
                "reminders": [
                    "DO NOT add any text labels such as 'TOPPER', 'BASE', or 'TOPPER(WITH LOGO)' in the image",
                    "THE SCULPTURE MUST MATCH EXACTLY WITH THE ORIGINAL IMAGE WITHOUT ANY MODIFICATIONS",
                    "DO NOT add any text, words, or written content to the sculpture",
                    "DO NOT add any logos, brand names, or company names unless explicitly provided by user",
                    "DO NOT interpret any visual elements as text or add text based on what you think you see",
                    "ONLY add text or logos if the user explicitly uploads them or mentions them in their request",
                    "CRITICAL: ALWAYS photograph from FRONT VIEW - camera directly in front, centered, head-on, NO side angles",
                    "CRITICAL: WIDE ANGLE FRAMING - show sculpture in center with 30-40% space around it, NOT zoomed in close-up",
                    "CRITICAL: EXACT COPY - The generated ice sculpture MUST be an EXACT REPLICA of the input image geometry. Do NOT change the shape, proportions, or details.",
                    "CRITICAL: FONT PRESERVATION - If there is text in the input image, the generated ice sculpture MUST use the EXACT SAME FONT STYLE. Do NOT change the font.",
                    "CRITICAL: NO HALLUCINATIONS - Do not add any elements that are not present in the input image."
                ]
            }          
        }

        if any(term in user_input_lower for term in ['snofilled', 'paper', 'colored']):
            logo_instructions = {
                "Snofilled": {
                    "effect": "Create a carved snow appearance inside the ice sculpture. The image should not be colored and should be engraved with visible depth inside the ice cube.",
                    "strict": "The logo must be embedded a few centimeters into the ice",
                    "processing": "Remove any background of the image before embedding it into the ice cube"
                },
                "Colored": {
                    "effect": "It should look like the ice is colored from the logo, not etched. The image appears as colored pigmentation embedded inside the ice.",
                    "strict": "The logo must be embedded a few centimeters into the ice",
                    "processing": "Remove any background of the image before embedding it into the ice cube"
                },
                "Paper": {
                    "effect": "It should look like a colored printed paper is frozen inside the ice cube. The logo should be colored, have a slight white outline, and a transparent background, and should be centered within the cube.",
                    "strict": "The logo must be embedded a few centimeters into the ice",
                    "processing": "Remove any background of the image before embedding it into the ice cube"
                }
            }
            effect_type = None
            if 'snofilled' in user_input_lower:
                effect_type = "Snofilled"
            elif 'paper' in user_input_lower:
                effect_type = "Paper"
            elif 'colored' in user_input_lower:
                effect_type = "Colored"
            
            if effect_type:
                image_generation_dict["logo_instructions"] = {
                    "task": "add the image into the sculpture",
                    "effect_type": effect_type,
                    "instructions": logo_instructions[effect_type],
                    "clarification": "If a blue image is provided, it is always ice and should be used as the ice sculpture, not as a logo. Blue images are never logos or image overlays—they are the ice.",
                    "ice_structure": "The ice sculpture must precisely match the input image, with 100% accuracy. Do not add, remove, or modify any elements.",
                    "composition_rules": "CRITICAL: Generate exactly ONE single image. Do NOT create a side-by-side comparison. Do NOT show the original logo next to the ice. The logo must be INSIDE the ice. The output must be a single view of the final ice cube."
                }

        image_generation_prompt = json.dumps(image_generation_dict, indent=2)
        if 'template_selected_message' in session:
            image_generation_prompt += f"\n\nNOTE: {session['template_selected_message']}"
            session.pop('template_selected_message', None)

        if user_aspect_ratio and user_aspect_ratio != "auto":
            aspect_ratio = user_aspect_ratio
        else:
            aspect_ratio = "16:9" if is_ice_bar else "9:16"
        
        image_prompts_json = request.form.get('image_prompts', '{}')
        try:
            image_prompts = json.loads(image_prompts_json)
        except json.JSONDecodeError:
            image_prompts = {}

        image_inputs = []
        for path in all_image_paths:
            if path and os.path.exists(path):
                try:
                    with Image.open(path) as img:
                        image_inputs.append(img.convert('RGB'))
                except Exception as e:
                    print(f"Error loading image {path}: {e}")

        try:
            contents = [image_generation_prompt]
            for i, img in enumerate(image_inputs):
                contents.append(img)
                specific_prompt = image_prompts.get(str(i), "").strip()
                if specific_prompt:
                    contents.append(f"Instructions for the image above (Image {i+1}): {specific_prompt}")
            
            if aspect_ratio:
                contents.append(f"Aspect Ratio: {aspect_ratio}")

            response = client2.models.generate_content(
                model=PRO_MODEL_ID,
                contents=contents,
                config=types.GenerateContentConfig(response_modalities=['Image'])
            )
            
            output_id = uuid.uuid4().hex[:8]
            output_path = os.path.join(app.config["GENERATED_FOLDER"], f"sculpture_{output_id}.png")
            for part in response.candidates[0].content.parts:
                if part.inline_data is not None:
                    pathlib.Path(output_path).write_bytes(part.inline_data.data)
                    optimize_saved_image(output_path)
                    break
            output_filename = os.path.basename(output_path)
            
            cloudinary_url = upload_and_save_generated_image(
                output_path, image_generation_prompt, "sculpture", output_filename,
                template_name=template_name, user_id=user_id
            )
            final_image_url = cloudinary_url if cloudinary_url else f"/static/generated/{output_filename}"

            session["conversation"].append({"role": "user", "content": user_input})
            session["conversation"].append({
                "role": "assistant", "content": "Here is your ice sculpture:", "image": final_image_url
            })
            session.modified = True
            
            shutil.rmtree(temp_dir)
            return jsonify({"image_url": final_image_url})
        except Exception as e:
            shutil.rmtree(temp_dir)
            return jsonify({"response": f"Error generating sculpture image: {str(e)}"})

    try:
        classification = classify_prompt_type(user_input)
        if classification == "text":
            system_prompt = f"""You are an AI assistant for an ice sculpture company..."""
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ]
            completion = client.chat.completions.create(
                model="gpt-4o", messages=messages
            )
            gpt_response = completion.choices[0].message.content
            session["conversation"].append({"role": "user", "content": user_input})
            session["conversation"].append({"role": "assistant", "content": gpt_response})
            return jsonify({"response": gpt_response})

        elif classification == "generate":
            generation_id = uuid.uuid4().hex[:8]
            if not image_generation_prompt:
                image_generation_prompt = f"""
        "task": "Generate realistic images of ice engravings based solely on user text input.",
        "user_input": {user_input}
        """
            response = client2.models.generate_content(
                model=PRO_MODEL_ID,
                contents=image_generation_prompt,
                config=types.GenerateContentConfig(response_modalities=['Image'])
            )

            with tempfile.TemporaryDirectory() as temp_dir:
                output_path = os.path.join(temp_dir, f"generated_{generation_id}.png")
                for part in response.candidates[0].content.parts:
                    if part.inline_data is not None:
                        pathlib.Path(output_path).write_bytes(part.inline_data.data)
                        optimize_saved_image(output_path)
                        break
                output_filename = os.path.basename(output_path)
                
                cloudinary_url = upload_and_save_generated_image(
                    output_path, image_generation_prompt, "text_to_image", output_filename,
                    template_name=template_name, user_id=user_id
                )
                final_image_url = cloudinary_url if cloudinary_url else f"/static/uploads/{output_filename}"

                session["conversation"].append({"role": "user", "content": user_input})
                session["conversation"].append({
                    "role": "assistant", "content": "Here is your ice sculpture:", "image": final_image_url
                })
                session.modified = True
                return jsonify({"response": "Here is your ice sculpture:", "image_url": final_image_url})

    except Exception as e:
        return jsonify({"response": f"Error: {str(e)}"})

# ---------------- Kling AI ----------------

KLING_ACCESS_KEY = os.getenv("KLING_ACCESS_KEY")
KLING_SECRET_KEY = os.getenv("KLING_SECRET_KEY")

def get_kling_token():
    headers = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "iss": KLING_ACCESS_KEY,
        "exp": int(time.time()) + 1800,
        "nbf": int(time.time()) - 5
    }
    return jwt.encode(payload, KLING_SECRET_KEY, algorithm="HS256", headers=headers)

@app.route('/api/create_video', methods=['POST'])
def create_video():
    try:
        data = request.get_json()
        image_url = data.get('image_url')
        prompt = data.get('prompt', 'show the ice sculpture/ice object from all angles, rotate the camera and show all sides of the sculpture.')
        
        if not image_url:
            return jsonify({'error': 'Missing image_url'}), 400
            
        if image_url.startswith('/static') or image_url.startswith('static'):
           base_dir = os.path.abspath(os.path.dirname(__file__))
           clean_path = image_url.replace('/static/', '').replace('static/', '')
           local_path = os.path.join(base_dir, 'static', clean_path)
           
           if os.path.exists(local_path):
               try:
                   upload_result = cloudinary.uploader.upload(local_path, folder="video_gen_input")
                   image_url = upload_result.get("secure_url")
               except Exception as e:
                   return jsonify({'error': f'Failed to upload local image for processing: {str(e)}'}), 500
           else:
               return jsonify({'error': 'Local image file not found'}), 404

        token = get_kling_token()
        url = "https://api-singapore.klingai.com/v1/videos/image2video"
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        payload = {
            "model_name": "kling-v1-6",
            "mode": "pro",
            "duration": "10",
            "image": image_url,
            "prompt": prompt,
            "cfg_scale": 0.5
        }
        
        response = requests.post(url, headers=headers, json=payload)
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify({'error': f"Kling API Error: {response.text}"}), response.status_code

    except Exception as e:
        return jsonify({'error': str(e)}), 500

def upload_and_save_generated_video(file_path, prompt, template_type, original_filename, user_id=None):
    try:
        upload_result = cloudinary.uploader.upload(
            file_path, folder="generated_videos", resource_type="video", public_id=os.path.splitext(original_filename)[0]
        )
        video_url = upload_result.get("secure_url")
        data = {
            "image_url": video_url,
            "prompt": prompt,
            "template_type": template_type,
            "original_filename": original_filename,
            "template_name": "Video Generation",
            "user_id": user_id
        }
        supabase.table("generated_images").insert(data).execute()
        
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception:
            pass
        return video_url
    except Exception as e:
        logging.error(f"Error uploading/saving generated video: {e}")
        return None

@app.route('/api/get_video_status/<task_id>', methods=['GET'])
def get_video_status(task_id):
    try:
        token = get_kling_token()
        url = f"https://api-singapore.klingai.com/v1/videos/image2video/{task_id}"
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            task_status = data.get('data', {}).get('task_status')
            
            if task_status == 'succeed':
                task_result = data.get('data', {}).get('task_result', {})
                videos = task_result.get('videos', [])
                if videos:
                    kling_video_url = videos[0].get('url')
                    if kling_video_url:
                        try:
                            video_response = requests.get(kling_video_url)
                            if video_response.status_code == 200:
                                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
                                    tmp_video.write(video_response.content)
                                    tmp_video_path = tmp_video.name
                                final_video_url = upload_and_save_generated_video(
                                    tmp_video_path, "Video generated from image", "video", f"video_{task_id}.mp4"
                                )
                                if final_video_url:
                                    data['data']['task_result']['videos'][0]['url'] = final_video_url
                        except Exception as e:
                            print(f"Error processing video download/upload: {e}")
            return jsonify(data)
        else:
            return jsonify({'error': f"Kling API Error: {response.text}"}), response.status_code
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/get_users', methods=['POST'])
def admin_get_users():
    try:
        user_id = get_current_user_id()
        if not user_id:
            return jsonify({'error': 'Unauthorized'}), 401
        
        if not supabase_admin:
             return jsonify({'error': 'Server misconfiguration: Admin client not available'}), 500

        current_user = supabase_admin.auth.admin.get_user_by_id(user_id)
        if not current_user or not current_user.user:
             return jsonify({'error': 'User not found'}), 404
        
        user_email = current_user.user.email
        if user_email not in ['alvaro@theicebutcher.com', 'alvaro@icebutcher.com']:
             return jsonify({'error': 'Forbidden: Admin access required'}), 403

        data = request.get_json() or {}
        page = data.get('page', 0)
        limit = data.get('limit', 50)
        
        users_response = supabase_admin.auth.admin.list_users(page=page + 1, per_page=limit)
        if isinstance(users_response, list):
            users = users_response
        else:
            users = users_response.users
        
        users_data = []
        for u in users:
            image_count = 0
            try:
                # Use supabase_admin to bypass RLS if necessary
                count_res = supabase_admin.table('generated_images').select('*', count='exact', head=True).eq('user_id', u.id).execute()
                image_count = count_res.count
            except Exception as e:
                logging.warning(f"Failed to fetch image count for user {u.id}: {e}")

            users_data.append({
                'image_count': image_count,
                'id': u.id,
                'email': u.email,
                'created_at': u.created_at,
                'last_sign_in_at': u.last_sign_in_at,
                'display_name': u.user_metadata.get('display_name') if u.user_metadata else None,
                'avatar_url': u.user_metadata.get('avatar_url') if u.user_metadata else None,
            })
            
        return jsonify({
            'data': users_data, 
            'total': users_response.total if hasattr(users_response, 'total') else len(users_data)
        }), 200

    except Exception as e:
        logging.error(f"Error in admin_get_users: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    if not os.path.exists(app.config["UPLOAD_FOLDER"]):
        os.makedirs(app.config["UPLOAD_FOLDER"])
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=debug)
