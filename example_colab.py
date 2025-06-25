#!/usr/bin/env python3
"""
Chatterbox TTS & VC - Google Colab Examples
Simple examples for programmatic usage in Colab
"""

import torch
import torchaudio as ta
from IPython.display import Audio, display
import os

# Device detection
if torch.cuda.is_available():
    device = "cuda"
    print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = "cpu"
    print("💻 Using CPU")

def example_tts_basic():
    """Basic TTS example"""
    print("\n🎤 Basic TTS Example")
    print("=" * 30)
    
    from chatterbox.tts import ChatterboxTTS
    
    # Load model
    print("Loading TTS model...")
    model = ChatterboxTTS.from_pretrained(device=device)
    
    # Generate speech
    text = "Hello! This is Chatterbox TTS running in Google Colab. The quality is amazing!"
    print(f"Generating: '{text}'")
    
    wav = model.generate(text)
    
    # Save and display
    output_path = "tts_basic_example.wav"
    ta.save(output_path, wav, model.sr)
    print(f"✅ Saved to: {output_path}")
    
    # Play in Colab
    display(Audio(output_path))
    
    return output_path

def example_tts_with_voice_cloning():
    """TTS with voice cloning example"""
    print("\n🎭 TTS with Voice Cloning Example")
    print("=" * 40)
    
    from chatterbox.tts import ChatterboxTTS
    
    # Load model
    print("Loading TTS model...")
    model = ChatterboxTTS.from_pretrained(device=device)
    
    # Note: You would need to upload a reference audio file
    # For this example, we'll use the default voice with different parameters
    text = "This is an example of expressive speech with emotion control!"
    
    print(f"Generating expressive speech: '{text}'")
    
    # Generate with higher exaggeration
    wav = model.generate(
        text,
        exaggeration=0.8,  # More expressive
        cfg_weight=0.3,    # Faster pace
        temperature=0.9    # More variation
    )
    
    # Save and display
    output_path = "tts_expressive_example.wav"
    ta.save(output_path, wav, model.sr)
    print(f"✅ Saved to: {output_path}")
    
    # Play in Colab
    display(Audio(output_path))
    
    return output_path

def example_voice_conversion():
    """Voice conversion example"""
    print("\n🔄 Voice Conversion Example")
    print("=" * 35)
    
    from chatterbox.vc import ChatterboxVC
    
    # Load model
    print("Loading VC model...")
    model = ChatterboxVC.from_pretrained(device=device)
    
    # Note: For a real example, you would need source and target audio files
    print("⚠️  Voice conversion requires audio files to be uploaded")
    print("📁 Upload your audio files and use:")
    print("   wav = model.generate('source.wav', target_voice_path='target.wav')")
    
    return None

def example_batch_generation():
    """Generate multiple TTS samples"""
    print("\n📦 Batch TTS Generation Example")
    print("=" * 40)
    
    from chatterbox.tts import ChatterboxTTS
    
    # Load model
    print("Loading TTS model...")
    model = ChatterboxTTS.from_pretrained(device=device)
    
    # Multiple texts with different styles
    examples = [
        {
            "text": "Welcome to our AI demonstration.",
            "exaggeration": 0.3,
            "cfg_weight": 0.7,
            "name": "formal"
        },
        {
            "text": "This is so exciting and amazing!",
            "exaggeration": 0.9,
            "cfg_weight": 0.3,
            "name": "excited"
        },
        {
            "text": "Once upon a time, in a land far away...",
            "exaggeration": 0.5,
            "cfg_weight": 0.6,
            "name": "narrative"
        }
    ]
    
    output_files = []
    
    for i, example in enumerate(examples):
        print(f"Generating {example['name']}: '{example['text']}'")
        
        wav = model.generate(
            example["text"],
            exaggeration=example["exaggeration"],
            cfg_weight=example["cfg_weight"]
        )
        
        output_path = f"batch_example_{example['name']}.wav"
        ta.save(output_path, wav, model.sr)
        output_files.append(output_path)
        
        print(f"✅ Saved: {output_path}")
        display(Audio(output_path))
    
    return output_files

def example_parameter_comparison():
    """Compare different parameter settings"""
    print("\n⚙️ Parameter Comparison Example")
    print("=" * 40)
    
    from chatterbox.tts import ChatterboxTTS
    
    # Load model
    print("Loading TTS model...")
    model = ChatterboxTTS.from_pretrained(device=device)
    
    text = "This is a test of different parameter settings."
    
    # Different parameter combinations
    configs = [
        {"exaggeration": 0.3, "cfg_weight": 0.7, "name": "calm_high_quality"},
        {"exaggeration": 0.5, "cfg_weight": 0.5, "name": "balanced"},
        {"exaggeration": 0.8, "cfg_weight": 0.3, "name": "expressive_fast"},
    ]
    
    print(f"Comparing parameters for: '{text}'")
    
    for config in configs:
        print(f"\n🎛️ {config['name']}: exaggeration={config['exaggeration']}, cfg_weight={config['cfg_weight']}")
        
        wav = model.generate(
            text,
            exaggeration=config["exaggeration"],
            cfg_weight=config["cfg_weight"]
        )
        
        output_path = f"param_comparison_{config['name']}.wav"
        ta.save(output_path, wav, model.sr)
        
        print(f"✅ Generated: {output_path}")
        display(Audio(output_path))

def main():
    """Run all examples"""
    print("🎙️ Chatterbox Examples for Google Colab")
    print("=" * 50)
    print("📱 These examples will generate audio files and play them in Colab")
    print("🔊 Make sure your audio is enabled!")
    print()
    
    try:
        # Run examples
        example_tts_basic()
        example_tts_with_voice_cloning()
        example_voice_conversion()
        example_batch_generation()
        example_parameter_comparison()
        
        print("\n🎉 All examples completed!")
        print("📁 Check the generated .wav files in your Colab environment")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        print("💡 Make sure you have installed chatterbox-tts: !pip install chatterbox-tts")

if __name__ == "__main__":
    main()
