#!/usr/bin/env python3
"""
Test script to verify Colab fixes work properly
"""

import torch
import warnings
warnings.filterwarnings("ignore")

def test_device_detection():
    """Test device detection"""
    print("🔧 Testing device detection...")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✅ Device detected: {device}")
    
    if device == "cuda":
        try:
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"✅ GPU: {gpu_name}")
            print(f"✅ Memory: {gpu_memory:.1f} GB")
        except Exception as e:
            print(f"⚠️  GPU info error (normal in some environments): {e}")
    
    return device

def test_gradio_import():
    """Test Gradio import and basic components"""
    print("\n🔧 Testing Gradio import...")
    
    try:
        import gradio as gr
        print(f"✅ Gradio version: {gr.__version__}")
        
        # Test basic components without 'info' parameter
        print("✅ Testing Audio component...")
        audio = gr.Audio(sources=["upload"], type="filepath", label="Test Audio")
        
        print("✅ Testing Slider component...")
        slider = gr.Slider(0.0, 1.0, value=0.5, label="Test Slider")
        
        print("✅ Testing Textbox component...")
        textbox = gr.Textbox(value="Test", label="Test Textbox")
        
        print("✅ All Gradio components work correctly")
        return True
        
    except Exception as e:
        print(f"❌ Gradio test failed: {e}")
        return False

def test_chatterbox_import():
    """Test Chatterbox import"""
    print("\n🔧 Testing Chatterbox import...")
    
    try:
        from chatterbox.tts import ChatterboxTTS
        print("✅ ChatterboxTTS import successful")
        
        from chatterbox.vc import ChatterboxVC
        print("✅ ChatterboxVC import successful")
        
        return True
        
    except Exception as e:
        print(f"❌ Chatterbox import failed: {e}")
        print("💡 Try: !pip install chatterbox-tts")
        return False

def test_model_loading(device):
    """Test model loading (without full download)"""
    print(f"\n🔧 Testing model loading on {device}...")
    
    try:
        from chatterbox.tts import ChatterboxTTS
        
        print("⏳ Attempting to load TTS model (this may take time on first run)...")
        # This will download models if not cached
        model = ChatterboxTTS.from_pretrained(device=device)
        print("✅ TTS model loaded successfully")
        
        # Test basic generation
        print("⏳ Testing basic generation...")
        text = "Hello, this is a test."
        wav = model.generate(text)
        
        if wav is not None and wav.numel() > 0:
            print("✅ Audio generation successful")
            print(f"✅ Audio shape: {wav.shape}")
            print(f"✅ Sample rate: {model.sr}")
            return True
        else:
            print("❌ Generated audio is empty")
            return False
            
    except Exception as e:
        print(f"❌ Model loading/generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🧪 Chatterbox Colab Fix Tests")
    print("=" * 40)
    
    # Test 1: Device detection
    device = test_device_detection()
    
    # Test 2: Gradio import
    gradio_ok = test_gradio_import()
    
    # Test 3: Chatterbox import
    chatterbox_ok = test_chatterbox_import()
    
    # Test 4: Model loading (only if imports work)
    model_ok = False
    if chatterbox_ok:
        model_ok = test_model_loading(device)
    
    # Summary
    print("\n📊 Test Summary")
    print("=" * 20)
    print(f"Device Detection: {'✅' if device else '❌'}")
    print(f"Gradio Import: {'✅' if gradio_ok else '❌'}")
    print(f"Chatterbox Import: {'✅' if chatterbox_ok else '❌'}")
    print(f"Model Loading: {'✅' if model_ok else '❌'}")
    
    if all([device, gradio_ok, chatterbox_ok, model_ok]):
        print("\n🎉 All tests passed! Colab setup is working correctly.")
        print("🚀 You can now run the Gradio app: python colab_gradio_app.py")
    else:
        print("\n⚠️  Some tests failed. Check the errors above.")
        if not chatterbox_ok:
            print("💡 Install Chatterbox: !pip install chatterbox-tts")
        if not gradio_ok:
            print("💡 Install Gradio: !pip install gradio")

if __name__ == "__main__":
    main()
