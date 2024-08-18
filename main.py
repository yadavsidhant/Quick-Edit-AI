import os
import torch
import whisper
import cv2
import numpy as np
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, AudioFileClip, ImageClip, concatenate_videoclips, CompositeAudioClip
from pydub import AudioSegment
from gtts import gTTS
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from PIL import Image, ImageDraw, ImageFont
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Function to download and load AI models
def load_models():
    whisper_model = whisper.load_model("base").to(device)

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-1.3B").to(device)
    text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

    return whisper_model, text_generator

# Function to transcribe video
def transcribe_video(video_path, whisper_model):
    result = whisper_model.transcribe(video_path)
    return result["text"], result["segments"]

# Function to generate script
def generate_script(text_generator, duration):
    prompt = ""
    generated_texts = text_generator(prompt, max_length=400, num_return_sequences=3)
    script = f"Welcome to our news report.\n\n{generated_texts}}\n\nThat concludes our news report. Thank you for watching."
    return script

# Function to generate AI voice
def generate_voice(script):
    tts = gTTS(text=script, lang='en')
    tts.save("ai_voice.mp3")
    return AudioFileClip("ai_voice.mp3")

# Function to create a title card with dynamically fitting text
def create_title_card(headline, size=(1280, 720), duration=5):
    img = Image.new('RGB', size, color='black')
    draw = ImageDraw.Draw(img)

    font_path = "./fonts/Oswald-Heavy.ttf"
    font_size = 100
    font = ImageFont.truetype(font_path, font_size)

    while True:
        text_width, text_height = draw.textsize(headline, font=font)
        if text_width <= size[0] - 40:
            break
        font_size -= 5
        font = ImageFont.truetype(font_path, font_size)

    x = (size[0] - text_width) // 2
    y = (size[1] - text_height) // 2

    draw.text((x, y), headline, font=font, fill='white')

    img.save('title_card.png')
    return ImageClip('title_card.png').set_duration(duration)

# Function to edit video
def edit_video(main_video_path, resource_videos, images, bg_music_path, script, duration, headline):
    main_video = VideoFileClip(main_video_path)

    resource_clips = [VideoFileClip(video).subclip(0, 5) for video in resource_videos]
    image_clips = [ImageClip(img).set_duration(5) for img in images]

    voice_clip = generate_voice(script)
    bg_music = AudioFileClip(bg_music_path).volumex(0.1)

    title_card = create_title_card(headline)

    # Create a list of clips to concatenate
    clips = [title_card]

    # Add resource clips and image clips alternately
    for i in range(max(len(resource_clips), len(image_clips))):
        if i < len(resource_clips):
            clips.append(resource_clips[i])
        if i < len(image_clips):
            clips.append(image_clips[i])

    # Add main video at the end
    clips.append(main_video.subclip(0, min(30, main_video.duration)))

    final_clip = concatenate_videoclips(clips)

    # Adjust the duration of the final clip
    final_clip = final_clip.subclip(0, min(duration, final_clip.duration))

    # Add voice-over and background music
    voice_clip = voice_clip.subclip(0, min(duration, voice_clip.duration))
    bg_music = bg_music.subclip(0, final_clip.duration)

    final_audio = CompositeAudioClip([voice_clip, bg_music])
    final_clip = final_clip.set_audio(final_audio)

    return final_clip

# API route to create news video
@app.route('/create_news_video', methods=['POST'])
def create_news_video_api():
    data = request.json
    main_video_path = data.get('main_video_path')
    resource_videos = data.get('resource_videos', [])
    images = data.get('images', [])
    bg_music_path = data.get('bg_music_path')
    duration = data.get('duration', 60)
    headline = data.get('headline', 'News Headline')

    whisper_model, text_generator = load_models()

    script = generate_script(text_generator, duration)

    final_clip = edit_video(main_video_path, resource_videos, images, bg_music_path, script, duration, headline)

    output_path = "final_news_video.mp4"
    final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")

    return jsonify({"status": "success", "output_path": output_path})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)