## Installation

1. Clone the repository:
``` sh
git clone https://github.com/Quickom2023/gemma4-inference
```

2. Open folder: 
``` sh
cd gemma4-inference
```

3. Install the dependencies:
``` sh
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

- For NVIDIA GPU:
``` sh
pip install torch==2.8.0+cu128 torchvision==0.23.0+cu128 torchaudio==2.8.0+cu128 --index-url https://download.pytorch.org/whl/cu128
```

4. Install Tesseract OCR (for the keyframe-OCR safety backstop):

The service reads on-screen text from video frames to catch political content the
model can't read at low resolution. This needs the Tesseract binary plus the
Vietnamese language pack (`pytesseract` is already installed via `requirements.txt`).
Without Tesseract the backstop just disables itself — the rest of the service still runs.

- Ubuntu / Debian:
``` sh
sudo apt update
sudo apt install -y tesseract-ocr tesseract-ocr-vie
```

- macOS (Homebrew) — the `tesseract-lang` formula includes Vietnamese:
``` sh
brew install tesseract tesseract-lang
```

Verify the binary and that the Vietnamese (`vie`) pack is present:
``` sh
tesseract --version
tesseract --list-langs   # should list: eng, vie
```