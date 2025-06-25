# 🎙️ Chatterbox TTS & Voice Conversion - Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bhootnaat22/chatterbox-colab/blob/master/chatterbox_colab.ipynb)

Welcome to the Google Colab version of **Chatterbox** - Resemble AI's open-source TTS and Voice Conversion model! This repository provides an easy-to-use Colab notebook with a live Gradio interface.

## 🚀 Quick Start

### Option 1: Use the Colab Notebook (Recommended)
1. Click the "Open in Colab" badge above
2. Run all cells in order
3. Wait for the Gradio interface to launch
4. Use the public URL to access the interface

### Option 2: Manual Setup
```python
# In a Colab cell:
!git clone https://github.com/bhootnaat22/chatterbox-colab.git
%cd chatterbox-colab
!python launch_colab.py
```

## ✨ Features

### 🎤 Text-to-Speech (TTS)
- **Zero-shot voice cloning** from reference audio
- **Emotion/exaggeration control** for expressive speech
- **Advanced sampling parameters** for fine-tuning
- **Real-time generation** with progress tracking
- **Built-in examples** to get started quickly

### 🔄 Voice Conversion (VC)
- **Convert any voice** to sound like another
- **High-quality audio processing** 
- **Simple upload interface**
- **Fast processing** on GPU

### 🌐 Live Gradio Interface
- **Public URL sharing** - access from anywhere
- **Tabbed interface** for TTS and VC
- **Progress indicators** and status updates
- **Mobile-friendly** responsive design
- **Error handling** with helpful messages

## 🎯 Usage Tips

### For Best TTS Results:
- **Reference Audio**: Use 3-10 seconds of clear, high-quality speech
- **Text Length**: Keep under 300 characters for optimal performance
- **Exaggeration**: 
  - `0.5` = Neutral, natural speech
  - `0.7-1.0` = More expressive, dramatic
  - `0.3` = Calmer, more subdued
- **CFG/Pace**: 
  - `0.5` = Balanced quality and speed
  - `0.3` = Faster, good for expressive speech
  - `0.7` = Slower, higher quality

### For Voice Conversion:
- **Source Audio**: Any clear speech audio
- **Target Voice**: Optional reference for specific voice characteristics
- **Quality**: Higher quality input = better output

## 🔧 Technical Details

### System Requirements
- **GPU**: Recommended (T4, V100, A100)
- **RAM**: 8GB+ recommended
- **Storage**: ~3GB for models and dependencies

### Performance
- **GPU Generation**: 10-30 seconds per TTS
- **CPU Generation**: 1-3 minutes per TTS  
- **Voice Conversion**: 5-15 seconds
- **First Run**: Additional time for model download (~2GB)

### Models Used
- **TTS Model**: Chatterbox 0.5B parameter model
- **Voice Encoder**: For speaker embeddings
- **S3 Tokenizer**: For speech token generation
- **Watermarking**: Built-in Perth watermarker

## 🛠️ Troubleshooting

### Common Issues

**❌ Out of Memory Error**
```
Solution: Runtime → Restart Runtime, then try shorter text
```

**❌ Model Download Failed**
```
Solution: Check internet connection, restart runtime
```

**❌ Audio Quality Issues**
```
Solution: Use higher quality reference audio (16kHz+, 3-10 seconds)
```

**❌ Gradio Interface Not Loading**
```
Solution: Wait for initialization, check public URL in output
```

### Performance Optimization
- Use GPU runtime for faster generation
- Keep text inputs concise
- Use high-quality reference audio
- Restart runtime if memory issues occur

## 📱 Interface Guide

### TTS Tab
1. **Enter Text**: Type or paste text to synthesize
2. **Upload Reference** (Optional): Add voice sample to clone
3. **Adjust Parameters**: Fine-tune emotion and quality
4. **Generate**: Click to create speech
5. **Download**: Save the generated audio

### VC Tab
1. **Upload Source**: Add audio to convert
2. **Upload Target** (Optional): Add target voice reference
3. **Convert**: Click to process
4. **Download**: Save the converted audio

## 🎵 Example Prompts

### Neutral Speech
```
"Hello, welcome to our presentation today. We'll be covering the latest developments in AI technology."
```

### Expressive Speech (use exaggeration ~0.8)
```
"This is absolutely incredible! I can't believe how amazing this technology is!"
```

### Narrative Style (use CFG ~0.7)
```
"In a world where artificial intelligence meets human creativity, new possibilities emerge every day."
```

## 🔗 Links & Resources

- 🏠 [Original Chatterbox Repository](https://github.com/resemble-ai/chatterbox)
- 🤗 [Hugging Face Space](https://huggingface.co/spaces/ResembleAI/Chatterbox)
- 🎵 [Demo Samples](https://resemble-ai.github.io/chatterbox_demopage/)
- 💬 [Discord Community](https://discord.gg/rJq9cRJBJ6)
- 🏢 [Resemble AI](https://resemble.ai)

## 📄 License

This project follows the same MIT license as the original Chatterbox repository.

## 🙏 Acknowledgments

- **Resemble AI** for the amazing Chatterbox model
- **Google Colab** for providing free GPU access
- **Gradio** for the excellent web interface framework
- **Hugging Face** for model hosting and distribution

## ⚠️ Responsible AI

This technology includes built-in watermarking and should be used responsibly:
- Don't create misleading or harmful content
- Respect voice ownership and consent
- Follow local laws and regulations
- Use for creative and educational purposes

---

### Made with ❤️ by the Community

*Enjoy creating amazing speech with Chatterbox in Google Colab!*
