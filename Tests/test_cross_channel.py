import sys
import os
import torch
import torchaudio
import soundfile as sf

# 1. FIX: APPLY MONKEYPATCHES BEFORE ANY OTHER IMPORTS
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]

# 2. THE NUCLEAR OPTION: Global override for loading audio
def custom_audio_load(filepath, channels_first=True, **kwargs):
    data, samplerate = sf.read(filepath, dtype='float32')
    tensor = torch.from_numpy(data)
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    else:
        tensor = tensor.t()
    if not channels_first:
        tensor = tensor.t()
    return tensor, samplerate

torchaudio.load = custom_audio_load

# 3. FIX: Add parent directory to path so 'src' is findable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 4. NOW it is safe to import SpeechBrain-dependent modules
from src.verification.ecapa_engine import EcapaVerifier

def main():
    verifier = EcapaVerifier()
    
    clean_file = "data/samples/Haris_Sample.wav"
    telephony_file = "data/samples/Haris_Sample_Rough_without.729.wav"

    if not os.path.exists(clean_file) or not os.path.exists(telephony_file):
        print(f"[!] Error: Missing files. Ensure {clean_file} and {telephony_file} exist.")
        return

    print(f"\n[RUNNING CROSS-CHANNEL VERIFICATION]")
    print(f"File 1 (Clean): {clean_file}")
    print(f"File 2 (8kHz Noisy): {telephony_file}")

    # The verify_pair method uses self.model.verify_files, 
    # which will now use our custom_audio_load automatically.
    score, is_match = verifier.verify_pair(clean_file, telephony_file)

    print(f"\n[RESULTS]")
    print(f"Similarity Score: {score:.4f}")
    print(f"Match Decision: {'MATCH' if is_match else 'NO MATCH'}")

if __name__ == "__main__":
    main()