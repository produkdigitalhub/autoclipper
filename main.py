import os
import json
import whisper
import ffmpeg
from google import genai

# 1. Konversi Detik ke Format Timestamp Subtitle ASS (H:MM:SS.cs)
def format_ass_time(seconds):
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    centiseconds = int(round((seconds - int(seconds)) * 100))
    if centiseconds >= 100:
        centiseconds = 99
    return f"{hours}:{minutes:02d}:{secs:02d}.{centiseconds:02d}"

# 2. Generate Subtitle ASS dengan Highlight Kata Aktiv
def generate_ass_subtitles(words, ass_output_path, clip_start_time):
    header = """[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: TikTokFont, Arial, 24, &H00FFFFFF, &H0000FFFF, &H00000000, &H00000000, 1, 0, 0, 0, 100, 100, 0, 0, 1, 5, 2, 2, 50, 50, 960, 1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = []
    chunk_size = 4  # Menampilkan 4 kata per grup di layar
    
    for i in range(0, len(words), chunk_size):
        chunk = words[i:i + chunk_size]
        if not chunk:
            continue
        
        for highlight_idx, active_word in enumerate(chunk):
            word_start = max(0, active_word['start'] - clip_start_time)
            word_end = max(0, active_word['end'] - clip_start_time)
            
            line_parts = []
            for idx, w in enumerate(chunk):
                word_text = w['word'].strip().upper()
                if idx == highlight_idx:
                    # Highlight kata aktif: Warna Kuning & Font Membesar
                    line_parts.append(f"{{\\c&H0000FFFF\\fscx115\\fscy115}}{word_text}{{\\rTikTokFont}}")
                else:
                    line_parts.append(f"{{\\c&H00FFFFFF}}{word_text}")
            
            formatted_text = " ".join(line_parts)
            start_str = format_ass_time(word_start)
            end_str = format_ass_time(word_end)
            
            events.append(f"Dialogue: 0,{start_str},{end_str},TikTokFont,,0,0,0,,{formatted_text}")

    with open(ass_output_path, "w", encoding="utf-8") as f:
        f.write(header + "\n".join(events))

# 3. Alur Otomatisasi Utama
def process_long_video(video_path, output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)
    
    print("Memuat Whisper Model...")
    whisper_model = whisper.load_model("small")
    
    print("Mengekstrak transkripsi & kata-kata...")
    result = whisper_model.transcribe(video_path, word_timestamps=True, verbose=False)
    
    all_words = []
    transcript_text = ""
    for segment in result['segments']:
        transcript_text += f"[{segment['start']:.1f}s - {segment['end']:.1f}s] {segment['text']}\n"
        if 'words' in segment:
            all_words.extend(segment['words'])

    print("Menganalisis segmen terbaik dengan Gemini AI...")
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    
    prompt = f"""
    Berikut adalah transkripsi video panjang. Pilih 3-5 bagian terbaik/viral berdurasi 30-60 detik.
    Format output HARUS JSON murni tanpa penjelas lain:
    [
        {{"part": 1, "start": 120.0, "end": 165.0, "title": "Topik Lucu"}},
        {{"part": 2, "start": 500.0, "end": 550.0, "title": "Highlight Utama"}}
    ]

    Transkripsi:
    {transcript_text}
    """

    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
    )
    
    clean_json = response.text.strip().replace("```json", "").replace("```", "")
    clips_data = json.loads(clean_json)

    for item in clips_data:
        part_num = item['part']
        start_time = item['start']
        end_time = item['end']
        
        print(f"Memproses Part {part_num}: {item['title']}...")
        
        clip_words = [w for w in all_words if start_time <= w['start'] <= end_time]
        ass_filename = os.path.join(output_dir, f"subtitles_part_{part_num}.ass")
        generate_ass_subtitles(clip_words, ass_filename, start_time)
        
        output_video = os.path.join(output_dir, f"clip_part_{part_num}.mp4")
        escaped_ass_path = ass_filename.replace("\\", "/").replace(":", "\\:")
        
        # Potong & tempel subtitle
        (
            ffmpeg
            .input(video_path, ss=start_time, to=end_time)
            .filter('subtitles', escaped_ass_path)
            .output(output_video, vcodec='libx264', acodec='aac')
            .overwrite_output()
            .run(quiet=True)
        )
        print(f"Selesai: {output_video}")

if __name__ == "__main__":
    video_input = "input.mp4"
    
    # Jika input.mp4 tidak ada, cari file .mp4 lain di direktori utama
    if not os.path.exists(video_input):
        mp4_files = [f for f in os.listdir('.') if f.endswith('.mp4')]
        if mp4_files:
            video_input = mp4_files[0]
            print(f"Meninggalkan pencarian default, menggunakan file yang ditemukan: {video_input}")
    
    if os.path.exists(video_input):
        process_long_video(video_input)
    else:
        print(f"Error: File video '{video_input}' tidak ditemukan di direktori!")
