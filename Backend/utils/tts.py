import shutil
import subprocess
from pathlib import Path

VOICES_DIR = Path(__file__).resolve().parent / "voices"
DEFAULT_MODEL = VOICES_DIR / "es_ES-sharvard-medium.onnx"


def _find_piper() -> str:
    return shutil.which("piper") or ""


def text_to_speech(
    text: str,
    output_path: str | Path,
    model: str | Path = DEFAULT_MODEL,
    length_scale: float = 1.1,
    noise_scale: float = 0.5,
) -> Path:
    piper = _find_piper()
    if not piper:
        raise FileNotFoundError("No se encontró piper")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    process = subprocess.run(
        [
            piper,
            "-m",
            str(model),
            "-f",
            str(output),
            "--length-scale",
            str(length_scale),
            "--noise-scale",
            str(noise_scale),
        ],
        input=text.encode("utf-8"),
        check=True,
        capture_output=True,
    )

    return output


def play_audio(audio_path: str | Path) -> None:
    aplay = shutil.which("aplay")
    if not aplay:
        print("Advertencia: aplay no está disponible; no se pudo reproducir el audio")
        return
    subprocess.run([aplay, "-q", str(audio_path)], check=False)