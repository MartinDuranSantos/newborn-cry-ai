import sys
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.tts import play_audio, text_to_speech

ENDPOINT_URL = "http://127.0.0.1:8000/clasificar"
DATASET_PATH = Path("/home/mds/PhD/MY_PROJECTS/infant-cri-ai/Dataset/archive/cry")
HUNGRY_FOLDER = "lonely"
SCARED_FOLDER = "scared"
TIRED_FOLDER = "tired"
SUPPORTED_FORMATS = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".3gp"}

FRASES_CATEGORIA = {
    "dolor de barriga": "El bebé tiene dolor de barriga",
    "eructos": "El bebé necesita eructar",
    "frío/calor": "El bebé siente frío, o siente calor",
    "malestar": "El bebé siente malestar",
    "hambre": "El bebé tiene hambre",
    "risa": "El bebé se está riendo",
    "soledad": "El bebé se siente solo",
    "ruido": "El bebé está incómodo por el ruido",
    "miedo": "El bebé tiene miedo",
    "silencio": "El bebé está en silencio",
    "cansancio": "El bebé tiene sueño",
}


def pick_hungry_audio() -> Path:
    folder = DATASET_PATH / HUNGRY_FOLDER
    if not folder.exists():
        sys.exit(f"Folder no existe: {folder}")

    audios = [f for f in sorted(folder.iterdir()) if f.suffix.lower() in SUPPORTED_FORMATS]
    if not audios:
        sys.exit(f"No hay audios válidos en {folder}")

    return audios[0]


def classify(audio_path: Path) -> dict:
    with open(audio_path, "rb") as f:
        response = requests.post(
            ENDPOINT_URL,
            files={"file": (audio_path.name, f, "audio/wav")},
            timeout=30,
        )

    if response.status_code != 200:
        sys.exit(f"Error {response.status_code}: {response.text}")

    return response.json()


def build_frase(categoria: str, confianza: float) -> str:
    frase = FRASES_CATEGORIA.get(categoria, f"La categoría es {categoria}")
    porcentaje = round(confianza * 100)
    return f"{frase}. La precisión del modelo es de {porcentaje} por ciento."


def main() -> None:
    audio_path = pick_hungry_audio()
    print(f"Enviando: {audio_path.name}")

    result = classify(audio_path)
    print("\n=== Resultado ===")
    print(f"Categoría:    {result['categoria']}")
    print(f"Confianza:    {result['confianza']:.4f}")
    print(f"Advertencia:  {result['advertencia']}")
    print(f"Mensaje:      {result['mensaje']}")

    frase = build_frase(result["categoria"], result["confianza"])
    print(f"\nVoz: {frase}")

    output = Path(__file__).resolve().parent / "audio_hambre.wav"
    text_to_speech(frase, output)
    print(f"Audio guardado en: {output}")

    play_audio(output)


if __name__ == "__main__":
    main()