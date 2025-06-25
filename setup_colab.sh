#!/bin/bash

# Chatterbox TTS & VC - Google Colab Setup Script
# This script sets up the environment for running Chatterbox in Google Colab

echo "🎙️ Chatterbox TTS & VC - Colab Setup"
echo "===================================="

# Update system packages
echo "📦 Updating system packages..."
apt-get update -qq
apt-get install -y -qq ffmpeg wget curl

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
pip install -q --upgrade pip
pip install -q chatterbox-tts gradio

# Download the Gradio app
echo "📱 Downloading Gradio app..."
wget -q https://raw.githubusercontent.com/bhootnaat22/chatterbox-colab/master/colab_gradio_app.py

# Download examples
echo "📚 Downloading examples..."
wget -q https://raw.githubusercontent.com/bhootnaat22/chatterbox-colab/master/example_colab.py

# Check installation
echo "✅ Checking installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import gradio; print(f'Gradio: {gradio.__version__}')"

# Check GPU availability
echo "🎯 Checking GPU availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}')"
    python -c "import torch; print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')"
fi

echo ""
echo "🚀 Setup complete! You can now:"
echo "   1. Run the Gradio app: python colab_gradio_app.py"
echo "   2. Try examples: python example_colab.py"
echo "   3. Use the Jupyter notebook cells"
echo ""
echo "📱 The Gradio app will provide a public URL for easy access"
echo "🎉 Enjoy using Chatterbox!"
