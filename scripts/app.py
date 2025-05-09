from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory
import os
import time
import uuid
import json
import logging
import traceback
import functools
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import torch
import torchvision
from torchvision import transforms
from PIL import Image
import mediapipe as mp

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("avatar_app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("AvatarApp")

app = Flask(__name__)

# App configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['DATA_FOLDER'] = 'data'
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png', 'mp4'}
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024  # 32MB max file size

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(os.path.join(app.config['DATA_FOLDER'], 'dataset'), exist_ok=True)
os.makedirs(os.path.join(app.config['DATA_FOLDER'], 'preprocessed'), exist_ok=True)
os.makedirs(os.path.join(app.config['DATA_FOLDER'], 'landmarks'), exist_ok=True)
os.makedirs(os.path.join(app.config['DATA_FOLDER'], 'embeddings'), exist_ok=True)
os.makedirs(os.path.join(app.config['DATA_FOLDER'], 'masks'), exist_ok=True)
os.makedirs(os.path.join(app.config['DATA_FOLDER'], 'llff'), exist_ok=True)

# Global model instances (initialize when needed)
segmentation_model = None
embedding_model = None
mp_face_mesh = None

def load_segmentation_model():
    """Load DeepLabV3+ model for segmentation"""
    global segmentation_model
    if segmentation_model is None:
        try:
            logger.info("Loading DeepLabV3+ segmentation model...")
            segmentation_model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            segmentation_model.to(device).eval()
            logger.info(f"Segmentation model loaded successfully (device: {device})")
        except Exception as e:
            logger.error(f"Error loading segmentation model: {e}")
            raise e
    return segmentation_model

def load_embedding_model():
    """Load DINOv2 or ResNet model for embedding extraction"""
    global embedding_model
    if embedding_model is None:
        try:
            logger.info("Loading DINOv2 model for embeddings...")
            try:
                embedding_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
            except Exception:
                logger.info("Falling back to ResNet50...")
                embedding_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
                # Remove the final classification layer
                embedding_model = torch.nn.Sequential(*list(embedding_model.children())[:-1])
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            embedding_model.to(device).eval()
            logger.info(f"Embedding model loaded successfully (device: {device})")
        except Exception as e:
            logger.error(f"Error loading embedding model: {e}")
            raise e
    return embedding_model

def load_face_mesh():
    """Load MediaPipe face mesh model"""
    global mp_face_mesh
    if mp_face_mesh is None:
        try:
            logger.info("Initializing MediaPipe Face Mesh...")
            mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5
            )
            logger.info("MediaPipe Face Mesh initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing MediaPipe Face Mesh: {e}")
            raise e
    return mp_face_mesh

def handle_exceptions(route_function):
    """Decorator to handle exceptions in routes."""
    @functools.wraps(route_function)
    def wrapper(*args, **kwargs):
        try:
            return route_function(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {route_function.__name__}: {str(e)}")
            logger.error(traceback.format_exc())
            return render_template('error.html', message=f"An error occurred: {str(e)}"), 500
    return wrapper

def allowed_file(filename):
    """Check if file has allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def process_image_segmentation(img_path, output_path):
    """Process a single image with segmentation"""
    try:
        # Load the image
        img = cv2.imread(img_path)
        if img is None:
            logger.error(f"Failed to load image: {img_path}")
            return False
        
        # Convert to PIL Image for processing
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        # Get model and device
        model = load_segmentation_model()
        device = next(model.parameters()).device
        
        # Preprocess image
        preprocess = transforms.Compose([
            transforms.Resize((480, 480)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        input_tensor = preprocess(pil_img).unsqueeze(0).to(device)
        
        # Generate segmentation mask
        with torch.no_grad():
            output = model(input_tensor)['out'][0]
            output_predictions = output.argmax(0).byte().cpu().numpy()
        
        # Extract person mask (COCO class 15)
        PERSON_CLASS = 15
        binary_mask = (output_predictions == PERSON_CLASS).astype(np.uint8) * 255
        
        # Improve mask quality
        kernel = np.ones((5, 5), np.uint8)
        binary_mask = cv2.dilate(binary_mask, kernel, iterations=1)
        binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        binary_mask = cv2.GaussianBlur(binary_mask, (7, 7), 0)
        
        # Resize mask to original image size
        h, w = img.shape[:2]
        binary_mask = cv2.resize(binary_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Create RGBA image with alpha channel as mask
        rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        rgba[:, :, 3] = binary_mask
        
        # Save output image
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, rgba)
        
        logger.info(f"Segmentation successful: {img_path} -> {output_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error segmenting image {img_path}: {e}")
        logger.error(traceback.format_exc())
        return False

def extract_landmarks(img_path, json_path):
    """Extract facial landmarks from an image and save as JSON"""
    try:
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            logger.error(f"Failed to load image: {img_path}")
            return False
        
        # Convert to RGB for MediaPipe
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get face mesh model
        face_mesh = load_face_mesh()
        
        # Process with MediaPipe
        results = face_mesh.process(rgb_img)
        
        # Check if face was detected
        if not results.multi_face_landmarks:
            logger.warning(f"No face detected in {img_path}")
            return False
        
        # Extract landmarks
        face_landmarks = []
        for idx, landmark in enumerate(results.multi_face_landmarks[0].landmark):
            face_landmarks.append({
                'idx': idx,
                'x': landmark.x,
                'y': landmark.y,
                'z': landmark.z
            })
        
        # Create landmark data structure
        landmark_data = {
            'face': face_landmarks,
            'image_info': {
                'height': img.shape[0],
                'width': img.shape[1],
                'channels': img.shape[2]
            }
        }
        
        # Save to JSON
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, 'w') as f:
            json.dump(landmark_data, f, indent=2)
        
        logger.info(f"Landmarks extracted: {img_path} -> {json_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error extracting landmarks from {img_path}: {e}")
        logger.error(traceback.format_exc())
        return False

def extract_embedding(img_path, npy_path):
    """Extract visual embedding from an image and save as NPY"""
    try:
        # Load image
        img = Image.open(img_path).convert('RGB')
        
        # Get model and device
        model = load_embedding_model()
        device = next(model.parameters()).device
        
        # Preprocess image
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # Generate embedding
        with torch.no_grad():
            embedding = model(img_tensor).cpu().numpy()[0]
        
        # Save embedding
        os.makedirs(os.path.dirname(npy_path), exist_ok=True)
        np.save(npy_path, embedding)
        
        logger.info(f"Embedding extracted: {img_path} -> {npy_path}")
        return True
    
    except Exception as e:
        logger.error(f"Error extracting embedding from {img_path}: {e}")
        logger.error(traceback.format_exc())
        return False

def generate_llff_data(subject_id):
    """Generate LLFF format data for a subject (placeholder implementation)"""
    try:
        logger.info(f"Generating LLFF data for subject {subject_id}")
        
        # Create LLFF directory
        llff_dir = os.path.join(app.config['DATA_FOLDER'], 'llff', subject_id)
        os.makedirs(llff_dir, exist_ok=True)
        
        # For now, create a dummy poses_bounds.npy file
        poses_bounds = np.zeros((1, 17))
        np.save(os.path.join(llff_dir, 'poses_bounds.npy'), poses_bounds)
        
        # Create train split
        with open(os.path.join(llff_dir, 'train_ids.txt'), 'w') as f:
            f.write('0\n')
        
        logger.info(f"LLFF data generation completed for subject {subject_id}")
        return True
    
    except Exception as e:
        logger.error(f"Error generating LLFF data for subject {subject_id}: {e}")
        logger.error(traceback.format_exc())
        return False

# def process_subject(subject_id):
#     """Process all data for a subject"""
#     try:
#         logger.info(f"Starting processing for subject {subject_id}")
        
#         # Set up directory paths
#         subject_dataset_dir = os.path.join(app.config['DATA_FOLDER'], 'dataset', subject_id)
#         subject_preprocessed_dir = os.path.join(app.config['DATA_FOLDER'], 'preprocessed', subject_id)
#         subject_landmarks_dir = os.path.join(app.config['DATA_FOLDER'], 'landmarks', subject_id)
#         subject_embeddings_dir = os.path.join(app.config['DATA_FOLDER'], 'embeddings', subject_id)
        
#         # Process each pose
#         poses_dir = os.path.join(subject_dataset_dir, 'poses')
#         if os.path.exists(poses_dir):
#             for pose in os.listdir(poses_dir):
#                 pose_dir = os.path.join(poses_dir, pose)
#                 if not os.path.isdir(pose_dir):
#                     continue
                
#                 # Create output directories for this pose
#                 preprocessed_pose_dir = os.path.join(subject_preprocessed_dir, 'poses', pose)
#                 landmarks_pose_dir = os.path.join(subject_landmarks_dir, 'poses', pose)
#                 embeddings_pose_dir = os.path.join(subject_embeddings_dir, 'poses', pose)
                
#                 os.makedirs(preprocessed_pose_dir, exist_ok=True)
#                 os.makedirs(landmarks_pose_dir, exist_ok=True)
#                 os.makedirs(embeddings_pose_dir, exist_ok=True)
                
#                 # Process images in this pose directory
#                 image_files = [f for f in os.listdir(pose_dir) 
#                              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
#                 logger.info(f"Processing {len(image_files)} images for {pose} pose")
                
#                 for img_file in image_files:
#                     img_path = os.path.join(pose_dir, img_file)
                    
#                     # Get base filename without extension
#                     base_name = os.path.splitext(img_file)[0]
                    
#                     # Define output paths
#                     preprocessed_path = os.path.join(preprocessed_pose_dir, f"{base_name}.png")
#                     landmarks_path = os.path.join(landmarks_pose_dir, f"{base_name}.json")
#                     embeddings_path = os.path.join(embeddings_pose_dir, f"{base_name}.npy")
                    
#                     # Skip if already processed
#                     if os.path.exists(preprocessed_path) and \
#                        os.path.exists(landmarks_path) and \
#                        os.path.exists(embeddings_path):
#                         logger.info(f"Skipping already processed image: {img_file}")
#                         continue
                    
#                     # Step 1: Segmentation
#                     if not os.path.exists(preprocessed_path):
#                         if not process_image_segmentation(img_path, preprocessed_path):
#                             logger.error(f"Segmentation failed for {img_file}, skipping further processing")
#                             continue
                    
#                     # Step 2: Landmark extraction (uses segmented image)
#                     if not os.path.exists(landmarks_path):
#                         extract_landmarks(preprocessed_path, landmarks_path)
                    
#                     # Step 3: Embedding extraction (uses segmented image)
#                     if not os.path.exists(embeddings_path):
#                         extract_embedding(preprocessed_path, embeddings_path)
        
#         # Generate LLFF data
#         generate_llff_data(subject_id)
        
#         logger.info(f"Processing completed for subject {subject_id}")
#         return True
    
#     except Exception as e:
#         logger.error(f"Error processing subject {subject_id}: {e}")
#         logger.error(traceback.format_exc())
#         return False


def process_subject(subject_id):
    """Process all data for a subject"""
    try:
        logger.info(f"Starting processing for subject {subject_id}")
        
        # Set up directory paths
        subject_dataset_dir = os.path.join(app.config['DATA_FOLDER'], 'dataset', subject_id)
        subject_preprocessed_dir = os.path.join(app.config['DATA_FOLDER'], 'preprocessed', subject_id)
        subject_landmarks_dir = os.path.join(app.config['DATA_FOLDER'], 'landmarks', subject_id)
        subject_embeddings_dir = os.path.join(app.config['DATA_FOLDER'], 'embeddings', subject_id)
        
        # Process each pose
        poses_dir = os.path.join(subject_dataset_dir, 'poses')
        if os.path.exists(poses_dir):
            for pose in os.listdir(poses_dir):
                pose_dir = os.path.join(poses_dir, pose)
                if not os.path.isdir(pose_dir):
                    continue
                
                # Create output directories for this pose
                preprocessed_pose_dir = os.path.join(subject_preprocessed_dir, 'poses', pose)
                landmarks_pose_dir = os.path.join(subject_landmarks_dir, 'poses', pose)
                embeddings_pose_dir = os.path.join(subject_embeddings_dir, 'poses', pose)
                
                os.makedirs(preprocessed_pose_dir, exist_ok=True)
                os.makedirs(landmarks_pose_dir, exist_ok=True)
                os.makedirs(embeddings_pose_dir, exist_ok=True)
                
                # Process images in this pose directory
                image_files = [f for f in os.listdir(pose_dir) 
                              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                logger.info(f"Processing {len(image_files)} images for {pose} pose")
                
                for img_file in image_files:
                    img_path = os.path.join(pose_dir, img_file)
                    
                    # Get base filename without extension
                    base_name = os.path.splitext(img_file)[0]
                    
                    # Define output paths
                    preprocessed_path = os.path.join(preprocessed_pose_dir, f"{base_name}.png")
                    landmarks_path = os.path.join(landmarks_pose_dir, f"{base_name}.json")
                    embeddings_path = os.path.join(embeddings_pose_dir, f"{base_name}.npy")
                    
                    # Skip if already processed
                    if os.path.exists(preprocessed_path) and \
                       os.path.exists(landmarks_path) and \
                       os.path.exists(embeddings_path):
                        logger.info(f"Skipping already processed image: {img_file}")
                        continue
                    
                    # Step 1: Segmentation
                    if not os.path.exists(preprocessed_path):
                        if not process_image_segmentation(img_path, preprocessed_path):
                            logger.error(f"Segmentation failed for {img_file}, skipping further processing")
                            continue
                    
                    # Step 2: Landmark extraction (uses segmented image)
                    if not os.path.exists(landmarks_path):
                        extract_landmarks(preprocessed_path, landmarks_path)
                    
                    # Step 3: Embedding extraction (uses segmented image)
                    if not os.path.exists(embeddings_path):
                        extract_embedding(preprocessed_path, embeddings_path)
        
        # Generate LLFF data
        generate_llff_data(subject_id)
        
        logger.info(f"Processing completed for subject {subject_id}")
        return True
    
    except Exception as e:
        logger.error(f"Error processing subject {subject_id}: {e}")
        logger.error(traceback.format_exc())
        return False

# def save_uploaded_files(subject_id, uploaded_files=None):
#     """Save uploaded files to the dataset directory"""
#     try:
#         logger.info(f"Saving uploaded files for subject {subject_id}")
        
#         # Available poses
#         poses = ['front', 'left', 'right', 'up', 'down']
        
#         # Create subject dataset directory
#         subject_dir = os.path.join(app.config['DATA_FOLDER'], 'dataset', subject_id)
#         poses_dir = os.path.join(subject_dir, 'poses')
#         os.makedirs(poses_dir, exist_ok=True)
        
#         # Check if we're using pose-specific uploads
#         pose_specific = False
#         for pose in poses:
#             pose_files = request.files.getlist(f'pose_{pose}')
#             if pose_files and pose_files[0].filename:
#                 pose_specific = True
#                 break
        
#         processed_poses = []
        
#         if pose_specific:
#             # Handle pose-specific uploads
#             for pose in poses:
#                 pose_files = request.files.getlist(f'pose_{pose}')
#                 if pose_files and pose_files[0].filename:
#                     pose_dir = os.path.join(poses_dir, pose)
#                     os.makedirs(pose_dir, exist_ok=True)
                    
#                     for i, file in enumerate(pose_files):
#                         if file and file.filename:
#                             filename = f"{i:03d}.jpg"
#                             file_path = os.path.join(pose_dir, filename)
#                             file.save(file_path)
                    
#                     processed_poses.append(pose)
#         else:
#             # Handle generic upload (all files to 'front' pose)
#             files_to_process = uploaded_files if uploaded_files else request.files.getlist('image_files')
            
#             if files_to_process and files_to_process[0].filename:
#                 front_dir = os.path.join(poses_dir, 'front')
#                 os.makedirs(front_dir, exist_ok=True)
                
#                 for i, file in enumerate(files_to_process):
#                     if file and file.filename:
#                         filename = f"{i:03d}.jpg"
#                         file_path = os.path.join(front_dir, filename)
#                         file.save(file_path)
                
#                 processed_poses.append('front')
        
#         logger.info(f"Files saved for poses: {processed_poses}")
#         return processed_poses
    
#     except Exception as e:
#         logger.error(f"Error saving uploads for subject {subject_id}: {e}")
#         logger.error(traceback.format_exc())
#         raise e


def save_uploaded_files(subject_id, uploaded_files=None):
    """Save uploaded files to the dataset directory with better handling of multiple files"""
    try:
        logger.info(f"Saving uploaded files for subject {subject_id}")
        
        # Available poses
        poses = ['front', 'left', 'right', 'up', 'down']
        
        # Create subject dataset directory
        subject_dir = os.path.join(app.config['DATA_FOLDER'], 'dataset', subject_id)
        poses_dir = os.path.join(subject_dir, 'poses')
        os.makedirs(poses_dir, exist_ok=True)
        
        # Check if we're using pose-specific uploads
        pose_specific = False
        for pose in poses:
            pose_files = request.files.getlist(f'pose_{pose}')
            if pose_files and pose_files[0].filename:
                pose_specific = True
                break
        
        processed_poses = []
        
        if pose_specific:
            # Handle pose-specific uploads
            for pose in poses:
                pose_files = request.files.getlist(f'pose_{pose}')
                if pose_files and pose_files[0].filename:
                    pose_dir = os.path.join(poses_dir, pose)
                    os.makedirs(pose_dir, exist_ok=True)
                    
                    # Get existing files to determine next index
                    existing_files = [f for f in os.listdir(pose_dir) 
                                     if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                    start_index = len(existing_files)
                    
                    for i, file in enumerate(pose_files):
                        if file and file.filename:
                            # Use incremental index for filenames
                            filename = f"{start_index + i:03d}.jpg"
                            file_path = os.path.join(pose_dir, filename)
                            file.save(file_path)
                    
                    processed_poses.append(pose)
        else:
            # Handle generic upload (all files to 'front' pose)
            files_to_process = uploaded_files if uploaded_files else request.files.getlist('image_files')
            
            if files_to_process and files_to_process[0].filename:
                front_dir = os.path.join(poses_dir, 'front')
                os.makedirs(front_dir, exist_ok=True)
                
                # Get existing files to determine next index
                existing_files = [f for f in os.listdir(front_dir) 
                                 if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                start_index = len(existing_files)
                
                for i, file in enumerate(files_to_process):
                    if file and file.filename:
                        # Use incremental index for filenames
                        filename = f"{start_index + i:03d}.jpg"
                        file_path = os.path.join(front_dir, filename)
                        file.save(file_path)
                
                processed_poses.append('front')
        
        logger.info(f"Files saved for poses: {processed_poses}")
        return processed_poses
    
    except Exception as e:
        logger.error(f"Error saving uploads for subject {subject_id}: {e}")
        logger.error(traceback.format_exc())
        raise e

@app.route('/')
@handle_exceptions
def index():
    return render_template('index.html')

@app.route('/webcam-capture')
@handle_exceptions
def webcam_capture():
    return render_template('webcam_capture.html')

@app.route('/upload-files')
@handle_exceptions
def upload_files():
    return render_template('upload_files.html')

@app.route('/api/upload-process', methods=['POST'])
@handle_exceptions
def upload_process():
    """Unified endpoint for uploading and processing files"""
    # Generate subject ID if not provided
    subject_id = request.form.get('subject_id', f'subject_{int(time.time())}')
    
    # Check for any files
    has_files = False
    for key in request.files:
        files = request.files.getlist(key)
        if files and files[0].filename:
            has_files = True
            break
    
    if not has_files:
        return jsonify({
            'status': 'error',
            'message': 'No files were uploaded'
        }), 400
    
    try:
        # Save uploaded files
        processed_poses = save_uploaded_files(subject_id)
        
        if not processed_poses:
            return jsonify({
                'status': 'error',
                'message': 'Failed to save any files'
            }), 400
        
        # Process the subject (synchronously)
        success = process_subject(subject_id)
        
        if success:
            return jsonify({
                'status': 'success',
                'subject_id': subject_id,
                'redirect': f'/results/{subject_id}'
            })
        else:
            return jsonify({
                'status': 'error',
                'message': 'Processing error occurred'
            }), 500
            
    except Exception as e:
        logger.error(f"Error in upload_process: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500
@app.route('/api/start-pipeline', methods=['POST'])
@handle_exceptions
def start_pipeline():
    """Legacy endpoint for the webcam capture page - redirects to upload_process"""
    return upload_process()
@app.route('/results/<subject_id>')
@handle_exceptions
def results(subject_id):
    # Check if results exist
    subject_dir = os.path.join(app.config['DATA_FOLDER'], 'preprocessed', subject_id)
    if not os.path.exists(subject_dir):
        return render_template('error.html', message=f"No results found for subject {subject_id}")
    
    # Get pose images
    poses_dir = os.path.join(subject_dir, 'poses')
    pose_images = {}
    
    if os.path.exists(poses_dir):
        for pose in ['front', 'left', 'right', 'up', 'down']:
            pose_dir = os.path.join(poses_dir, pose)
            if os.path.exists(pose_dir):
                images = [f for f in os.listdir(pose_dir) if f.endswith('.png')]
                pose_images[pose] = images
    
    # Get action videos if any
    actions_dir = os.path.join(subject_dir, 'actions')
    action_frames = {}
    
    if os.path.exists(actions_dir):
        for action in os.listdir(actions_dir):
            action_dir = os.path.join(actions_dir, action)
            if os.path.isdir(action_dir):
                frames = [f for f in os.listdir(action_dir) if f.endswith('.png')]
                action_frames[action] = len(frames)
    
    # Get landmark info
    landmark_dir = os.path.join(app.config['DATA_FOLDER'], 'landmarks', subject_id)
    has_landmarks = os.path.exists(landmark_dir)
    
    # Get embedding info
    embedding_dir = os.path.join(app.config['DATA_FOLDER'], 'embeddings', subject_id)
    has_embeddings = os.path.exists(embedding_dir)
    
    # Get LLFF data info
    llff_dir = os.path.join(app.config['DATA_FOLDER'], 'llff', subject_id)
    has_llff_data = os.path.exists(os.path.join(llff_dir, "poses_bounds.npy"))
    
    # Get train/val split counts if they exist
    train_count = 0
    val_count = 0
    
    if has_llff_data:
        train_ids_path = os.path.join(llff_dir, "train_ids.txt")
        val_ids_path = os.path.join(llff_dir, "val_ids.txt")
        
        if os.path.exists(train_ids_path):
            with open(train_ids_path, 'r') as f:
                train_count = len(f.readlines())
        
        if os.path.exists(val_ids_path):
            with open(val_ids_path, 'r') as f:
                val_count = len(f.readlines())
    
    return render_template(
        'results.html',
        subject_id=subject_id,
        pose_images=pose_images,
        action_frames=action_frames,
        has_landmarks=has_landmarks,
        has_embeddings=has_embeddings,
        has_llff_data=has_llff_data,
        train_count=train_count,
        val_count=val_count
    )

@app.route('/data/<path:filename>')
@handle_exceptions
def serve_data(filename):
    """Serve files from the data directory."""
    return send_from_directory(app.config['DATA_FOLDER'], filename)

@app.route('/static/<path:filename>')
@handle_exceptions
def serve_static(filename):
    """Serve static files."""
    return send_from_directory('static', filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)