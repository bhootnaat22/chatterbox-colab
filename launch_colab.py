#!/usr/bin/env python3
"""
Quick launcher for Chatterbox in Google Colab
This script handles installation and launches the Gradio interface
"""

import subprocess
import sys
import os

def install_dependencies():
    """Install required dependencies"""
    print("🔄 Installing dependencies...")
    
    # System dependencies
    subprocess.run(["apt-get", "update", "-qq"], check=False)
    subprocess.run(["apt-get", "install", "-y", "-qq", "ffmpeg"], check=False)
    
    # Python dependencies
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "chatterbox-tts", "gradio"], check=True)
    
    print("✅ Dependencies installed!")

def check_environment():
    """Check if we're running in Colab"""
    try:
        import google.colab
        return True
    except ImportError:
        return False

def download_app():
    """Download the Gradio app if not present"""
    if not os.path.exists("colab_gradio_app.py"):
        print("📥 Downloading Gradio app...")
        subprocess.run([
            "wget", "-q", 
            "https://raw.githubusercontent.com/bhootnaat22/chatterbox-colab/master/colab_gradio_app.py"
        ], check=True)
        print("✅ App downloaded!")

def main():
    """Main launcher function"""
    print("🎙️ Chatterbox TTS & VC - Colab Launcher")
    print("=" * 50)
    
    # Check environment
    is_colab = check_environment()
    if is_colab:
        print("✅ Running in Google Colab")
    else:
        print("⚠️  Not running in Google Colab - some features may not work")
    
    # Install dependencies
    install_dependencies()
    
    # Download app
    download_app()
    
    # Launch the app
    print("🚀 Launching Chatterbox...")
    print("📱 This may take a few minutes on first run")
    print("🌐 Look for the public URL in the output below")
    print("=" * 50)
    
    # Import and run the app
    try:
        exec(open("colab_gradio_app.py").read())
    except Exception as e:
        print(f"❌ Error launching app: {e}")
        print("💡 Try running the cells in the notebook manually")

if __name__ == "__main__":
    main()
