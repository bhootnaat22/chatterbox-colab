#!/usr/bin/env python3
"""
Chatterbox TTS & Voice Conversion - Google Colab Gradio App
Optimized for Google Colab with combined TTS and VC interface
"""

import os
import random
import numpy as np
import torch
import gradio as gr
from chatterbox.tts import ChatterboxTTS
from chatterbox.vc import ChatterboxVC
import warnings
warnings.filterwarnings("ignore")

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🎯 Using device: {DEVICE}")

# Suppress CUDA warnings that are common in Colab
if DEVICE == "cuda":
    print("ℹ️  Note: CUDA warnings above are normal in Colab and can be ignored")

# Global model states
tts_model = None
vc_model = None

def set_seed(seed: int):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def load_tts_model():
    """Load TTS model with error handling"""
    global tts_model
    try:
        if tts_model is None:
            print("🔄 Loading TTS model...")
            tts_model = ChatterboxTTS.from_pretrained(DEVICE)
            print("✅ TTS model loaded successfully!")
        return tts_model
    except Exception as e:
        print(f"❌ Error loading TTS model: {e}")
        return None

def load_vc_model():
    """Load VC model with error handling"""
    global vc_model
    try:
        if vc_model is None:
            print("🔄 Loading VC model...")
            vc_model = ChatterboxVC.from_pretrained(DEVICE)
            print("✅ VC model loaded successfully!")
        return vc_model
    except Exception as e:
        print(f"❌ Error loading VC model: {e}")
        return None

def generate_tts(text, audio_prompt_path, exaggeration, temperature, seed_num,
                cfgw, min_p, top_p, repetition_penalty, progress=gr.Progress()):
    """Generate TTS with progress tracking and quality optimization"""
    try:
        progress(0.1, desc="Loading TTS model...")
        model = load_tts_model()
        if model is None:
            return None, "❌ Failed to load TTS model"

        progress(0.3, desc="Setting up generation...")
        if seed_num != 0:
            set_seed(int(seed_num))

        # Validate text length
        if len(text.strip()) == 0:
            return None, "❌ Please enter some text to synthesize"
        if len(text) > 300:
            return None, "❌ Text too long! Please keep it under 300 characters"

        # Optimize parameters for better quality
        # Clamp values to safe ranges
        exaggeration = max(0.25, min(2.0, exaggeration))
        temperature = max(0.1, min(1.5, temperature))  # Limit temperature for stability
        cfgw = max(0.1, min(1.0, cfgw))  # Ensure CFG is not 0
        min_p = max(0.01, min(0.2, min_p))  # Keep min_p in reasonable range

        progress(0.5, desc="Generating speech...")

        # Generate with optimized settings
        wav = model.generate(
            text,
            audio_prompt_path=audio_prompt_path,
            exaggeration=exaggeration,
            temperature=temperature,
            cfg_weight=cfgw,
            min_p=min_p,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )

        # Ensure audio is properly formatted
        if wav is None or wav.numel() == 0:
            return None, "❌ Generated audio is empty"

        # Convert to numpy and ensure proper shape
        audio_np = wav.squeeze().detach().cpu().numpy()

        # Basic audio validation
        if len(audio_np.shape) > 1:
            audio_np = audio_np[0]  # Take first channel if stereo

        # Normalize audio to prevent clipping
        if audio_np.max() > 1.0 or audio_np.min() < -1.0:
            audio_np = audio_np / max(abs(audio_np.max()), abs(audio_np.min()))

        progress(1.0, desc="Complete!")
        return (model.sr, audio_np), "✅ Generation successful!"

    except Exception as e:
        error_msg = f"❌ Generation failed: {str(e)}"
        print(f"TTS Error: {e}")
        import traceback
        traceback.print_exc()
        return None, error_msg

def generate_vc(audio, target_voice_path, progress=gr.Progress()):
    """Generate voice conversion with progress tracking and quality optimization"""
    try:
        progress(0.1, desc="Loading VC model...")
        model = load_vc_model()
        if model is None:
            return None, "❌ Failed to load VC model"

        if audio is None:
            return None, "❌ Please upload an audio file to convert"

        progress(0.5, desc="Converting voice...")
        wav = model.generate(audio, target_voice_path=target_voice_path)

        # Ensure audio is properly formatted
        if wav is None or wav.numel() == 0:
            return None, "❌ Generated audio is empty"

        # Convert to numpy and ensure proper shape
        audio_np = wav.squeeze().detach().cpu().numpy()

        # Basic audio validation
        if len(audio_np.shape) > 1:
            audio_np = audio_np[0]  # Take first channel if stereo

        # Normalize audio to prevent clipping
        if audio_np.max() > 1.0 or audio_np.min() < -1.0:
            audio_np = audio_np / max(abs(audio_np.max()), abs(audio_np.min()))

        progress(1.0, desc="Complete!")
        return (model.sr, audio_np), "✅ Voice conversion successful!"

    except Exception as e:
        error_msg = f"❌ Voice conversion failed: {str(e)}"
        print(f"VC Error: {e}")
        import traceback
        traceback.print_exc()
        return None, error_msg

# Custom CSS for better appearance
custom_css = """
.gradio-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.main-header {
    text-align: center;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 20px;
    border-radius: 10px;
    margin-bottom: 20px;
}
.status-box {
    padding: 10px;
    border-radius: 5px;
    margin: 10px 0;
}
.success { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
.error { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
.info { background-color: #d1ecf1; color: #0c5460; border: 1px solid #bee5eb; }
"""

# Create the Gradio interface
with gr.Blocks(css=custom_css, title="Chatterbox TTS & VC - Colab") as demo:
    
    # Header
    gr.HTML("""
    <div class="main-header">
        <h1>🎙️ Chatterbox TTS & Voice Conversion</h1>
        <p>Resemble AI's Open Source TTS and Voice Conversion - Running in Google Colab</p>
    </div>
    """)
    
    # Device info
    device_info = f"<strong>🎯 Device:</strong> {DEVICE.upper()} | <strong>🔥 PyTorch:</strong> {torch.__version__}"

    if DEVICE == "cuda":
        try:
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            gpu_name = torch.cuda.get_device_name(0)
            device_info += f" | <strong>🚀 GPU:</strong> {gpu_name} | <strong>💾 Memory:</strong> {gpu_memory:.1f} GB"
        except:
            device_info += " | <strong>🚀 GPU:</strong> Available"

    gr.HTML(f"""
    <div class="status-box info">
        {device_info}
        <br><small>ℹ️ CUDA warnings above are normal in Colab and can be ignored</small>
    </div>
    """)
    
    with gr.Tabs():
        # TTS Tab
        with gr.TabItem("🎤 Text-to-Speech", id="tts"):
            gr.Markdown("### Convert text to natural-sounding speech with emotion control")
            
            with gr.Row():
                with gr.Column(scale=2):
                    text_input = gr.Textbox(
                        value="Hello! This is Chatterbox TTS running in Google Colab. The voice quality is amazing, and I can control emotions too!",
                        label="📝 Text to Synthesize (max 300 characters)",
                        placeholder="Enter your text here...",
                        max_lines=4
                    )
                    gr.Markdown("💡 **Tip:** Use clear, well-punctuated text for best results")

                    ref_audio = gr.Audio(
                        sources=["upload", "microphone"],
                        type="filepath",
                        label="🎵 Reference Audio (Optional - Upload 3-10 seconds of clear speech)"
                    )
                    
                    with gr.Row():
                        exaggeration = gr.Slider(
                            0.25, 2.0, step=0.05, value=0.5,
                            label="🎭 Exaggeration (0.5=neutral, higher=more dramatic)"
                        )
                        cfg_weight = gr.Slider(
                            0.1, 1.0, step=0.05, value=0.7,
                            label="⚡ CFG/Pace (higher=slower but better quality)"
                        )
                    
                    with gr.Accordion("🔧 Advanced Settings", open=False):
                        with gr.Row():
                            seed_num = gr.Number(
                                value=0, label="🎲 Random Seed (0 for random)"
                            )
                            temperature = gr.Slider(
                                0.1, 1.5, step=0.05, value=0.7,
                                label="🌡️ Temperature (randomness)"
                            )
                        with gr.Row():
                            min_p = gr.Slider(
                                0.01, 0.20, step=0.01, value=0.08,
                                label="📊 Min P (newer sampler)"
                            )
                            top_p = gr.Slider(
                                0.00, 1.00, step=0.01, value=1.00,
                                label="🔝 Top P (1.0 disables)"
                            )
                        repetition_penalty = gr.Slider(
                            1.00, 2.00, step=0.1, value=1.2,
                            label="🔄 Repetition Penalty (prevents repetition)"
                        )
                    
                    tts_button = gr.Button("🎵 Generate Speech", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    tts_output = gr.Audio(label="🔊 Generated Speech", interactive=False)
                    tts_status = gr.HTML("")
            
            # TTS Examples
            gr.Markdown("### 💡 Quick Examples")
            with gr.Row():
                gr.Examples(
                    examples=[
                        ["Welcome to the future of AI-generated speech!", None, 0.5, 0.7, 0, 0.7, 0.08, 1.0, 1.2],
                        ["This is so exciting! I can't believe how realistic this sounds!", None, 0.7, 0.6, 0, 0.5, 0.08, 1.0, 1.2],
                        ["In a world where technology meets creativity, anything is possible.", None, 0.4, 0.7, 0, 0.8, 0.08, 1.0, 1.2],
                    ],
                    inputs=[text_input, ref_audio, exaggeration, temperature, seed_num, cfg_weight, min_p, top_p, repetition_penalty],
                    label="Click to try these examples"
                )
        
        # VC Tab  
        with gr.TabItem("🔄 Voice Conversion", id="vc"):
            gr.Markdown("### Convert one voice to sound like another")
            
            with gr.Row():
                with gr.Column():
                    source_audio = gr.Audio(
                        sources=["upload", "microphone"],
                        type="filepath",
                        label="🎤 Source Audio (upload the audio you want to convert)"
                    )
                    target_audio = gr.Audio(
                        sources=["upload", "microphone"],
                        type="filepath",
                        label="🎯 Target Voice (Optional - reference for target voice)"
                    )
                    vc_button = gr.Button("🔄 Convert Voice", variant="primary", size="lg")
                
                with gr.Column():
                    vc_output = gr.Audio(label="🔊 Converted Audio", interactive=False)
                    vc_status = gr.HTML("")
    
    # Footer
    gr.HTML("""
    <div style="text-align: center; margin-top: 20px; padding: 20px; background-color: #f8f9fa; border-radius: 10px;">
        <p><strong>🎉 Enjoying Chatterbox?</strong></p>
        <p>
            <a href="https://github.com/resemble-ai/chatterbox" target="_blank">⭐ Star on GitHub</a> | 
            <a href="https://discord.gg/rJq9cRJBJ6" target="_blank">💬 Join Discord</a> | 
            <a href="https://resemble.ai" target="_blank">🏠 Resemble AI</a>
        </p>
        <p><em>Made with ❤️ by Resemble AI • Use responsibly!</em></p>
    </div>
    """)
    
    # Event handlers
    tts_button.click(
        fn=generate_tts,
        inputs=[text_input, ref_audio, exaggeration, temperature, seed_num, 
                cfg_weight, min_p, top_p, repetition_penalty],
        outputs=[tts_output, tts_status],
        show_progress=True
    )
    
    vc_button.click(
        fn=generate_vc,
        inputs=[source_audio, target_audio],
        outputs=[vc_output, vc_status],
        show_progress=True
    )

if __name__ == "__main__":
    print("🚀 Starting Chatterbox Gradio App...")
    print("📱 This may take a few minutes on first run (downloading models)")
    
    # Launch with public sharing enabled for Colab
    demo.queue(
        max_size=20,
        default_concurrency_limit=2,
    ).launch(
        share=True,  # Enable public URL for Colab
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        quiet=False,
        debug=False
    )
