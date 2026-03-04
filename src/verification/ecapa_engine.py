import torchaudio
from speechbrain.inference.speaker import SpeakerRecognition
import os

class EcapaVerifier:
    def __init__(self, model_source="speechbrain/spkrec-ecapa-voxceleb"):
        print(f"Loading ECAPA-TDNN from {model_source}...")
        self.model = SpeakerRecognition.from_hparams(
            source=model_source,
            savedir=f"models/pretrained/{model_source.split('/')[-1]}"
        )
    def extract_embedding(self, filepath):
        """Extracts the raw 192-dimensional voice print for bulk testing."""
        signal, fs = torchaudio.load(filepath)
        
        # Ensure 16kHz for ECAPA
        if fs != 16000:
            signal = torchaudio.functional.resample(signal, orig_freq=fs, new_freq=16000)
            
        # Move audio to the same device as the model (CPU/GPU)
        signal = signal.to(self.model.device)
        
        with torch.no_grad():
            embeddings = self.model.encode_batch(signal)
            
        return embeddings[0, 0].cpu().numpy()
    
    def verify_pair(self, path1, path2, threshold=0.25):
        """
        Compares two audio files. 
        Note: threshold 0.25 is a common starting point for ECAPA-Voxceleb.
        """
        if not os.path.exists(path1) or not os.path.exists(path2):
            return None, "File(s) not found."

        score, prediction = self.model.verify_files(path1, path2)
        
        # prediction is based on the model's internal threshold
        # score is the raw cosine similarity
        return score.item(), prediction.item()

if __name__ == "__main__":
    # Quick Test logic
    verifier = EcapaVerifier()
    print("Verifier initialized and ready.")