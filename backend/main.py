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
import warnings # <--- Add this

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
    
    print("✅ All models loaded! API is ready to receive traffic.")
    yield
    # 2. Shutdown
    print("🛑 Shutting down server and clearing memory...")
    models.clear()

app = FastAPI(title="Telephony Speaker Verification API", lifespan=lifespan)

# --- ENDPOINTS ---

@app.get("/health")
async def health_check():
    return {"status": "online", "models_loaded": len(models) > 0}

@app.get("/challenge")
async def get_challenge(background_tasks: BackgroundTasks):
    """Generates a random 3-digit challenge and returns the audio prompt."""
    try:
        challenge_data = models["challenge_gen"].generate_numeric_challenge(length=3)
        # Note: In a real production app, you might stitch audio here and return the file.
        # For the API, returning the sequence is fastest.
        return {"challenge_sequence": challenge_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/enroll")
async def enroll_user(
    user_id: str = Form(...),
    take_1: UploadFile = File(...),
    take_2: UploadFile = File(...),
    take_3: UploadFile = File(...)
):
    """Creates a secure Multi-Template Dictionary (.pt) for a new user."""
    temp_files = []
    try:
        os.makedirs("data/temp", exist_ok=True)
        
        # Save uploads to temp files
        for i, file in enumerate([take_1, take_2, take_3]):
            temp_path = f"data/temp/{user_id}_take_{i}.wav"
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            temp_files.append(temp_path)
            
        # Run Enrollment (This takes a few seconds)
        success = models["enrollment_mgr"].create_multi_template(user_id, temp_files)
        
        if success:
            return {"status": "success", "message": f"User {user_id} securely enrolled."}
        else:
            raise HTTPException(status_code=500, detail="Enrollment processing failed.")
            
    finally:
        for f in temp_files:
            cleanup_temp_file(f)

@app.post("/verify")
async def verify_user(
    background_tasks: BackgroundTasks,
    user_id: str = Form(...),
    expected_challenge: str = Form(...), # e.g. "2,4,5"
    audio_file: UploadFile = File(...)
):
    """The Ultimate Gauntlet: Pyannote -> ASR -> ECAPA"""
    master_template = f"data/enrollments/{user_id}_baseline.pt"
    if not os.path.exists(master_template):
        raise HTTPException(status_code=404, detail="User not enrolled.")

    temp_audio_path = f"data/temp/live_{user_id}.wav"
    os.makedirs("data/temp", exist_ok=True)
    
    with open(temp_audio_path, "wb") as buffer:
        shutil.copyfileobj(audio_file.file, buffer)
        
    background_tasks.add_task(cleanup_temp_file, temp_audio_path)
    
    # Parse the expected challenge
    expected_sequence = [int(x.strip()) for x in expected_challenge.split(",")]

    try:
        # STAGE 1: Gatekeeper
        is_live, msg = models["gatekeeper"].check_audio_security(temp_audio_path)
        if not is_live:
            return JSONResponse(status_code=403, content={"status": "denied", "reason": msg})
            
        # STAGE 2: ASR Check
        transcription = models["asr"].transcribe(temp_audio_path)
        validation = models["validator"].evaluate_challenge(expected_sequence, transcription)
        
        if not validation["liveness_passed"]:
            return JSONResponse(status_code=403, content={
                "status": "denied", 
                "reason": validation["status_message"],
                "confidence": validation["confidence_score"]
            })
            
        # STAGE 3: ECAPA
        try: master_dict = torch.load(master_template, weights_only=False)
        except: master_dict = torch.load(master_template)
            
        emb_clean = np.array(master_dict["clean"].detach().cpu().numpy()).flatten()
        emb_telephony = np.array(master_dict["telephony"].detach().cpu().numpy()).flatten()
        emb_live = np.array(models["ecapa"].extract_embedding(temp_audio_path)).flatten()

        score = max(cosine_sim(emb_clean, emb_live), cosine_sim(emb_telephony, emb_live))
        
        if score >= 0.2393:
            return {"status": "granted", "score": float(score), "message": "Identity Verified."}
        else:
            return JSONResponse(status_code=403, content={"status": "denied", "reason": "Biometric mismatch.", "score": float(score)})
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))