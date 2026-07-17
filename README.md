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