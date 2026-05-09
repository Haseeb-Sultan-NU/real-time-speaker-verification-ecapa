import os
import shutil
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import torch
import torchaudio
import soundfile as sf
from dotenv import load_dotenv
import warnings
from pydub import AudioSegment
import uuid
from fastapi.responses import FileResponse
from fastapi import BackgroundTasks, HTTPException
import io
import base64
from pydantic import BaseModel
from passlib.context import CryptContext


from sqlalchemy.orm import Session
from fastapi import Depends
from src.database import get_db
from src.models import User, Enrollment, AuditLog

# --- SUPPRESS THIRD-PARTY WARNINGS ---
warnings.filterwarnings("ignore", category=UserWarning, message=".*torchaudio._backend.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*TypedStorage is deprecated.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*AudioMetaData.*")
warnings.filterwarnings("ignore", module="pyannote.*")
warnings.filterwarnings("ignore", module="speechbrain.*")

# --- WINDOWS MONKEYPATCHES ---
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]

def custom_audio_load(filepath, channels_first=True, **kwargs):
    data, samplerate = sf.read(filepath, dtype='float32')
    tensor = torch.from_numpy(data)
    if tensor.ndim == 1: tensor = tensor.unsqueeze(0)
    else: tensor = tensor.t()
    if not channels_first: tensor = tensor.t()
    return tensor, samplerate
torchaudio.load = custom_audio_load

# --- IMPORTS ---
from src.verification.gatekeeper import SecurityGatekeeper # Make sure this matches your class name!
from src.verification.digit_asr import UrduASRInference
from src.verification.liveness_validator import LivenessValidator
from src.verification.ecapa_engine import EcapaVerifier
from src.verification.challenge_generator import ChallengeGenerator
from src.verification.enrollment import EnrollmentManager
import numpy as np

# --- GLOBAL AI MODELS ---
models = {}
audio_cache = {}

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def cleanup_temp_file(filepath: str):
    """Background task to delete temporary audio files"""
    if os.path.exists(filepath):
        os.remove(filepath)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Boot up the Server & Models
    print("🚀 FASTAPI STARTUP: Loading AI Models into VRAM...")
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    
    # Load everything into the global dictionary
    models["gatekeeper"] = SecurityGatekeeper(hf_token)
    models["asr"] = UrduASRInference(model_size="base")
    models["validator"] = LivenessValidator(pass_threshold=0.80)
    models["ecapa"] = EcapaVerifier(finetuned_weights_path="models/best_urdu_triplet_ecapa.pth")
    models["challenge_gen"] = ChallengeGenerator()
    models["enrollment_mgr"] = EnrollmentManager("models/best_urdu_triplet_ecapa.pth")

    # --- PRELOAD URDU DIGITS INTO RAM ---
    print("🎙️ Caching Urdu IVR Audio Digits into Memory...")
    try:
        for i in range(10):
            file_path = f"data/audio_digits/{i}.wav"
            if os.path.exists(file_path):
                audio_cache[str(i)] = AudioSegment.from_wav(file_path)
            else:
                print(f"⚠️ WARNING: Missing audio file {file_path}")
        # Let's also add a half-second silence gap so the numbers don't blend together
        audio_cache["silence"] = AudioSegment.silent(duration=500) 
        print("✅ Audio assets cached successfully!")
    except Exception as e:
        print(f"❌ Audio Caching Failed: {e}")
    
    print("✅ All models loaded! API is ready to receive traffic.")
    yield
    # 2. Shutdown
    print("🛑 Shutting down server and clearing memory...")
    models.clear()

app = FastAPI(title="Telephony Speaker Verification API", lifespan=lifespan)

# --- SECURITY CONFIGURATION ---
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

# --- PYDANTIC SCHEMAS (For React JSON Payloads) ---
class UserSignup(BaseModel):
    user_id: str
    email: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str
# --- ENDPOINTS ---

@app.get("/health")
async def health_check():
    return {"status": "online", "models_loaded": len(models) > 0}

@app.get("/challenge")
async def get_challenge():
    try:
        # 1. Generate sequence
        challenge_data = models["challenge_gen"].generate_numeric_challenge(length=3)
        challenge_str = ",".join(map(str, challenge_data))
        
        # 2. Stitch in memory using BytesIO (No hard drive needed!)
        combined_audio = AudioSegment.empty()
        for digit in challenge_data:
            combined_audio += audio_cache[str(digit)]
            combined_audio += audio_cache["silence"]
        
        # 3. Export to a buffer instead of a file
        buffer = io.BytesIO()
        combined_audio.export(buffer, format="wav")
        
        # 4. Encode to Base64 string
        audio_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        print(f"✅ Challenge {challenge_str} generated and encoded.")

        # 5. Return JSON (Atomic response)
        return {
            "challenge_sequence": challenge_str,
            "audio_data": f"data:audio/wav;base64,{audio_base64}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/enroll")
async def enroll_user(
    user_id: str = Form(...),
    take_1: UploadFile = File(...),
    take_2: UploadFile = File(...),
    take_3: UploadFile = File(...),
    db: Session = Depends(get_db) # <--- INJECTING THE DATABASE HERE
):
    """Creates a secure Multi-Template Dictionary (.pt) and registers the user in PostgreSQL."""
    
    # 1. Check if user already has an active enrollment
    existing_enrollment = db.query(Enrollment).filter(Enrollment.user_id == user_id).first()
    if existing_enrollment:
        raise HTTPException(status_code=400, detail="User is already enrolled.")

    temp_files = []
    try:
        os.makedirs("data/temp", exist_ok=True)
        os.makedirs("data/enrollments", exist_ok=True)
        
        for i, file in enumerate([take_1, take_2, take_3]):
            temp_path = f"data/temp/{user_id}_take_{i}.wav"
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            temp_files.append(temp_path)
            
        # Run AI Enrollment
        success = models["enrollment_mgr"].create_multi_template(user_id, temp_files)
        
        if success:
            # 2. Database Transaction
            template_path = f"data/enrollments/{user_id}_baseline.pt"
            
            # Ensure User exists in the Users table first
            user = db.query(User).filter(User.user_id == user_id).first()
            if not user:
                user = User(user_id=user_id)
                db.add(user)
                db.flush() # Flushes to get the user ID without fully committing yet
                
            # Create the Enrollment record
            new_enrollment = Enrollment(user_id=user_id, template_uri=template_path)
            db.add(new_enrollment)
            
            db.commit() # Save everything securely!
            
            return {"status": "success", "message": f"User {user_id} securely enrolled and saved to database."}
        else:
            raise HTTPException(status_code=500, detail="Enrollment processing failed.")
            
    finally:
        for f in temp_files:
            cleanup_temp_file(f)


@app.post("/verify")
async def verify_user(
    background_tasks: BackgroundTasks,
    user_id: str = Form(...),
    expected_challenge: str = Form(...),
    audio_file: UploadFile = File(...),
    db: Session = Depends(get_db) # <--- DATABASE INJECTED
):
    """The Ultimate Gauntlet: Pyannote -> ASR -> ECAPA (Fully Audited)"""
    
    # 1. Database Check: Is this a valid, enrolled, unlocked user?
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found.")
    if user.locked_out:
        raise HTTPException(status_code=403, detail="Account is locked due to multiple failed attempts.")
        
    enrollment = db.query(Enrollment).filter(Enrollment.user_id == user_id).first()
    if not enrollment:
        raise HTTPException(status_code=404, detail="User has no biometric enrollment.")

    master_template = enrollment.template_uri # Get the path dynamically from PostgreSQL!
    if not os.path.exists(master_template):
        raise HTTPException(status_code=500, detail="Database error: Biometric template file missing from disk.")

    temp_audio_path = f"data/temp/live_{user_id}.wav"
    with open(temp_audio_path, "wb") as buffer:
        shutil.copyfileobj(audio_file.file, buffer)
    background_tasks.add_task(cleanup_temp_file, temp_audio_path)
    
    expected_sequence = [int(x.strip()) for x in expected_challenge.split(",")]

    # Prepare an Audit Log Record
    log_entry = AuditLog(
        user_id=user_id, 
        expected_challenge=expected_challenge,
        liveness_passed=False # Default to false until proven otherwise
    )

    try:
        # STAGE 1: Gatekeeper (Coercion Check)
        is_live, msg = models["gatekeeper"].check_audio_security(temp_audio_path)
        if not is_live:
            log_entry.coercion_detected = True
            log_entry.status = "DENIED_COERCION"
            db.add(log_entry)
            db.commit()
            return JSONResponse(status_code=403, content={"status": "denied", "reason": msg})
            
        # STAGE 2: ASR Check (Active Liveness)
        transcription = models["asr"].transcribe(temp_audio_path)
        log_entry.transcribed_text = transcription
        validation = models["validator"].evaluate_challenge(expected_sequence, transcription)
        
        if not validation["liveness_passed"]:
            log_entry.status = "DENIED_LIVENESS"
            db.add(log_entry)
            db.commit()
            return JSONResponse(status_code=403, content={"status": "denied", "reason": validation["status_message"]})
            
        log_entry.liveness_passed = True
            
        # STAGE 3: ECAPA (Biometric Match)
        try: master_dict = torch.load(master_template, weights_only=False)
        except: master_dict = torch.load(master_template)
            
        emb_clean = np.array(master_dict["clean"].detach().cpu().numpy()).flatten()
        emb_telephony = np.array(master_dict["telephony"].detach().cpu().numpy()).flatten()
        emb_live = np.array(models["ecapa"].extract_embedding(temp_audio_path)).flatten()

        score = float(max(cosine_sim(emb_clean, emb_live), cosine_sim(emb_telephony, emb_live)))
        log_entry.biometric_score = score
        
        if score >= 0.2393:
            log_entry.status = "GRANTED"
            
            # Reset failed attempts on success
            user.failed_attempts = 0
            db.add(user)
            db.add(log_entry)
            db.commit()
            return {"status": "granted", "score": score, "message": "Identity Verified."}
        else:
            log_entry.status = "DENIED_BIOMETRIC"
            
            # Increment failed attempts
            user.failed_attempts += 1
            if user.failed_attempts >= 3:
                user.locked_out = True
                
            db.add(user)
            db.add(log_entry)
            db.commit()
            return JSONResponse(status_code=403, content={"status": "denied", "reason": "Biometric mismatch.", "score": score})
            
    except Exception as e:
        db.rollback() # If python crashes, rollback the database
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/signup")
async def signup(user_data: UserSignup, db: Session = Depends(get_db)):
    """Registers a new user with a securely hashed password."""
    # Check if user ID or Email already exists
    if db.query(User).filter(User.user_id == user_data.user_id).first():
        raise HTTPException(status_code=400, detail="Username already taken.")
    if db.query(User).filter(User.email == user_data.email).first():
        raise HTTPException(status_code=400, detail="Email already registered.")

    # Hash the password and save
    hashed_pw = get_password_hash(user_data.password)
    new_user = User(
        user_id=user_data.user_id, 
        email=user_data.email, 
        hashed_password=hashed_pw
    )
    
    db.add(new_user)
    db.commit()
    
    return {"status": "success", "message": f"User {user_data.user_id} created successfully. Ready for biometric enrollment."}

@app.post("/login")
async def login(credentials: UserLogin, db: Session = Depends(get_db)):
    """Authenticates a user via email and password."""
    user = db.query(User).filter(User.email == credentials.email).first()
    
    # Verify existence and password
    if not user or not verify_password(credentials.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid email or password.")
        
    if user.locked_out:
        raise HTTPException(status_code=403, detail="Account is locked due to multiple failed biometric attempts.")

    # In a full production app, we would return a JWT Token here. 
    # For the FYP MVP, returning the user_id is perfectly sufficient for the React frontend to establish session state.
    return {
        "status": "success", 
        "user_id": user.user_id,
        "message": "Login successful"
    }