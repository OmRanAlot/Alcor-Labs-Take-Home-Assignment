# VLM Orchestrator - Setup Guide

Quick reference for getting your development environment ready.

## Quick Setup (30 seconds)

### GitHub Codespaces (Recommended)

1. Click **Code** → **Codespaces** → **Create codespace on main**
2. Wait for container to build (2-3 minutes)
3. In the terminal:
   ```bash
   export OPENROUTER_API_KEY=your_api_key_here
   make setup
   make dry-run
   ```

Done! Your environment is ready.

---

## Detailed Setup

### Option A: GitHub Codespaces

**Pros**: No local setup needed, pre-configured Python 3.11, automatic dependency installation.

**Steps**:

1. Go to the repository on GitHub
2. Click the **Code** button (green)
3. Select **Codespaces** tab
4. Click **Create codespace on main**
5. Wait for the container to initialize (watch the terminal at the bottom)
6. Once ready, open a terminal (Terminal → New Terminal if not visible)

**Configure API Key**:
```bash
# Set your OpenRouter API key (get it from https://openrouter.ai)
export OPENROUTER_API_KEY=your_key_here
```

**Verify Setup**:
```bash
make setup
make dry-run
```

If you see ✓ checkmarks, you're ready to go!

### Option B: Local Setup (macOS/Linux)

**Requirements**: Python 3.11+, pip, git, FFmpeg

**Install FFmpeg** (needed for audio extraction):
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg

# Verify
ffmpeg -version
```

**Steps**:

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/realtime-vlm-playground.git
   cd realtime-vlm-playground
   ```

2. Create a virtual environment:
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   make install
   ```

4. Set your API key:
   ```bash
   export OPENROUTER_API_KEY=your_api_key_here
   ```

5. Verify setup:
   ```bash
   make dry-run
   ```

### Option C: Local Setup (Windows)

**Requirements**: Python 3.11+ (from python.org), pip, git, FFmpeg

**Install FFmpeg**: Download from https://ffmpeg.org/download.html and add to your PATH.

**Steps**:

1. Clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/realtime-vlm-playground.git
   cd realtime-vlm-playground
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set your API key:
   ```bash
   set OPENROUTER_API_KEY=your_api_key_here
   ```

5. Verify setup:
   ```bash
   python src/run.py --dry-run
   ```

---

## Obtaining an OpenRouter API Key

1. Go to https://openrouter.ai
2. Sign up or log in
3. Go to your **API Keys** page (https://openrouter.ai/keys)
4. Create a new key (or use an existing one)
5. Copy the key
6. Set it in your environment:
   ```bash
   export OPENROUTER_API_KEY=sk_...your_key...
   ```

Monitor your usage on the [OpenRouter dashboard](https://openrouter.ai/activity).

Check [OpenRouter models](https://openrouter.ai/models) for the full catalog and current pricing.

---

## Verifying Your Setup

Run the dry-run test to validate everything is installed:

```bash
make dry-run
```

Expected output:
```
============================================================
  VLM ORCHESTRATOR
============================================================

  Procedure: Change Circuit Breaker (11 steps)
  Video:     data/videos_full/R066-15July-Circuit-Breaker-part2/Export_py/Video_pitchshift.mp4
  Speed:     1.0x

  [DRY RUN] Inputs validated. Skipping pipeline.
```

If you see the dry-run message without errors, you're ready!

---

## Troubleshooting

### "Python 3.11 not found"

**macOS** (using Homebrew):
```bash
brew install python@3.11
python3.11 -m venv venv
```

**Ubuntu/Debian**:
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv
python3.11 -m venv venv
```

**Windows**: Download from https://www.python.org/downloads/ and select Python 3.11

### "OPENROUTER_API_KEY not set"

Make sure you've exported the key:
```bash
export OPENROUTER_API_KEY=your_key  # macOS/Linux
set OPENROUTER_API_KEY=your_key     # Windows
```

Verify it's set:
```bash
echo $OPENROUTER_API_KEY  # macOS/Linux
echo %OPENROUTER_API_KEY%  # Windows
```

### "ffmpeg: command not found" or no audio in pipeline

FFmpeg is required for audio extraction. Install it:
```bash
brew install ffmpeg          # macOS
sudo apt install ffmpeg      # Ubuntu/Debian
```

### "opencv-python-headless installation fails"

If you're on an ARM Mac (M1/M2), use:
```bash
pip install opencv-python-headless --no-binary opencv-python-headless
```

Or use the pre-built version:
```bash
pip install opencv-python
```

### "ModuleNotFoundError: No module named 'src'"

Make sure you're running from the repository root:
```bash
cd /path/to/vlm-orchestrator-eval
python src/run.py --dry-run
```

### "Video file not found"

Download the training videos from [Google Drive](https://drive.google.com/drive/folders/1SDgpLC154P0nw2jQmknmgH5J9lLEieb5?usp=sharing) and unzip into the repo root:
```bash
unzip videos.zip   # extracts to data/videos_full/
```

---

## Development Workflow

```bash
# Activate your environment
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install development tools
make setup

# Run tests as you code
make test

# Format your code
make fmt

# Check for issues
make lint

# Run the pipeline
make run

# Clean up
make clean
```

---

## GitHub Codespaces Tips

**Secrets**: If you want to keep your API key secure in Codespaces:

1. Go to your GitHub repo Settings → Secrets and variables → Codespaces
2. Add a new secret: `OPENROUTER_API_KEY`
3. Set the value to your API key
4. In the Codespaces terminal, it will be available as an environment variable

**Persisting Changes**: All your changes are automatically saved to the remote branch.

**Stopping Codespaces**: To save credits, stop your Codespace when not in use:
- Bottom of VS Code → **Codespaces** → **Stop**

---

## Next Steps

Once setup is complete:

1. Read the main [`README.md`](../README.md) for the full assignment
2. Review `src/run.py` to understand what needs to be implemented
3. Check out the clip procedures: `data/clip_procedures/`
4. Review the utility modules: `src/data_loader.py` and `src/evaluator.py`
5. Start building your pipeline!

---

## Getting Help

If something isn't working:

1. **Check this guide** - scroll up!
2. **Check the main README** - has FAQ section
3. **Review error messages** carefully - they often say what's wrong
4. **Google the error** - stack overflow usually has answers
5. **Contact support** - reach out to the team

---

**Last Updated**: March 2026
