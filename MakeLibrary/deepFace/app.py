from flask import Flask, render_template, request, jsonify
import cv2
import os
from deepface import DeepFace
import tempfile

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_video():
    if 'video' not in request.files:
        return jsonify({'error': 'Tidak ada video yang diunggah'}), 400

    video_file = request.files['video']
    
    # Simpan video sementara
    temp_dir = tempfile.gettempdir()
    video_path = os.path.join(temp_dir, 'recorded_video.webm')
    video_file.save(video_path)

    # Buka video dengan OpenCV
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    os.remove(video_path)

    if len(frames) == 0:
        return jsonify({'error': 'Gagal membaca video atau video kosong.'}), 400

    # Ambil sampel ~60 frame secara merata untuk diproses (menghemat waktu komputasi)
    step = max(1, len(frames) // 60)
    sampled_frames = frames[::step][:60]

    emotion_totals = {
        'angry': 0, 'disgust': 0, 'fear': 0, 'happy': 0,
        'sad': 0, 'surprise': 0, 'neutral': 0
    }
    valid_frames = 0

    for frame in sampled_frames:
        try:
            # enforce_detection=False mencegah error jika wajah tidak terdeteksi pada frame tertentu
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            
            # DeepFace bisa mengembalikan list jika ada beberapa wajah, ambil wajah pertama
            if isinstance(result, list):
                result = result[0]
                
            emotions = result['emotion']
            for emo, val in emotions.items():
                emotion_totals[emo] += val
            valid_frames += 1
        except Exception:
            continue # Abaikan frame jika analisis gagal sama sekali

    if valid_frames == 0:
        return jsonify({'error': 'Tidak ada wajah yang dapat dianalisis dari video.'}), 400

    # Hitung rata-rata
    avg_emotions = {emo: (total / valid_frames) for emo, total in emotion_totals.items()}
    
    # Normalisasi agar totalnya mutlak 100%
    total_sum = sum(avg_emotions.values())
    final_percentages = {emo: round((val / total_sum) * 100, 2) for emo, val in avg_emotions.items()}

    # Cari emosi dominan
    dominant_emotion = max(final_percentages, key=final_percentages.get)

    return jsonify({
        'dominant': dominant_emotion,
        'percentages': final_percentages
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)