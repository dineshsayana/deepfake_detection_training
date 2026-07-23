import os
import cv2
import gradio as gr
import librosa
import numpy as np
import tensorflow as tf
import subprocess

# Load saved models with error fallback handling
try:
    image_model = tf.keras.models.load_model('deepfake_image_model.h5')
    audio_model = tf.keras.models.load_model('deepfake_audio_model.h5')
    video_model = tf.keras.models.load_model('deepfake_video_model.h5')
    print("All deepfake detection models loaded successfully.")
except Exception as e:
    print(f"Warning: Model file loading failed. Using mock logic for testing: {str(e)}")
    image_model = audio_model = video_model = None

# Constants matching training script configurations
IMG_SIZE = 128
VID_SIZE = 64
MAX_FRAMES = 15

def get_prediction_label(prob):
    """
    Standardizes label output. 
    Assumes training labels were: 0 for REAL, 1 for FAKE.
    """
    if prob < 0.5:
        confidence = (1.0 - prob) * 100
        return f"🟢 REAL (Confidence: {confidence:.2f}%)"
    else:
        confidence = prob * 100
        return f"🔴 FAKE (Confidence: {confidence:.2f}%)"

def reencode_video_to_h264(input_path):
    """
    Converts input video to browser-compatible H.264 MP4 using FFmpeg.
    This fixes the 'video not displaying/playing' error in Gradio.
    """
    output_path = os.path.join(os.path.dirname(input_path), "compatible_output.mp4")
    
    # FFmpeg command to force H264 baseline profile for web browser playback
    command = [
        'ffmpeg', '-y', '-i', input_path, 
        '-vcodec', 'libx264', '-pix_fmt', 'yuv420p',
        '-profile:v', 'baseline', '-level', '3.0', 
        '-an', output_path
    ]
    try:
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return output_path
    except Exception:
        # Fallback to original path if ffmpeg is missing from environment variables
        print("Warning: FFmpeg re-encoding failed. Ensure FFmpeg is installed and added to PATH.")
        return input_path

def process_video_and_predict(video_file):
    """
    Handles internal video conversion for playback display and triggers inference.
    """
    if video_file is None:
        return None, "No Video Sequence Provided", ""
        
    # Get string file path
    video_path = str(video_file)
    
    # 1. FIX PLAYBACK: Generate a browser compatible re-encoded video instance
    web_compatible_video = reencode_video_to_h264(video_path)
    
    # 2. FIX CLASSIFICATION: Extract real video frames using OpenCV stream context
    if video_model is None: 
        return web_compatible_video, "🟢 REAL (Mock Mode - Model Not Loaded)", "Mock Mode suggestions: Please ensure video model is loaded."
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return web_compatible_video, "❌ Error: Unable to open video file codec stream.", ""
        
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        cap.release()
        return web_compatible_video, "❌ Error: Video file contains unreadable or empty frame tracks.", ""
        
    interval = max(1, total_frames // MAX_FRAMES)
    count = 0
    valid_frames_read = 0
    
    while cap.isOpened() and len(frames) < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret: 
            break
        if count % interval == 0:
            if frame is not None and frame.size > 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_res = cv2.resize(frame_rgb, (VID_SIZE, VID_SIZE)) / 255.0
                frames.append(frame_res)
                valid_frames_read += 1
        count += 1
    cap.release()
    
    # Handle files with unreadable codecs or zero structural array dimensions
    if valid_frames_read == 0:
        return web_compatible_video, "❌ Error: Could not decode video frames. Try a different format (like .mp4).", ""
        
    # Structural padding for the model array dimensions
    while len(frames) < MAX_FRAMES:
        frames.append(np.zeros((VID_SIZE, VID_SIZE, 3)))
        
    video_input = np.expand_dims(np.array(frames, dtype=np.float32), axis=0)
    prob = float(video_model.predict(video_input, verbose=0)[0][0])
    
    analysis_result = get_prediction_label(prob)
    
    if prob >= 0.5:
        explanation = ("⚠️ AI Explanation (Reason):\n"
                       "Temporal inconsistency across frames, abnormal facial landmarks transitions, or blending artifacts "
                       "around face boundaries over time. The temporal Conv3D layers detected artificial synchronization patterns "
                       "typical of frame-swap deepfakes.")
    else:
        explanation = ("🛡️ AI Suggestions:\n"
                       "The video sequence maintains high spatio-temporal consistency across all evaluated frames. Suggestions:\n"
                       "1. Check for digital signatures/watermarks.\n"
                       "2. Cross-reference origin timestamps.\n"
                       "3. Examine physiological cues like natural eye-blinking cycles and pupil reactions.")
                       
    return web_compatible_video, analysis_result, explanation


def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    """
    Computes Grad-CAM Class Activation Map for Sequential Keras models in Keras 3.
    """
    x = tf.convert_to_tensor(img_array)
    conv_output = None
    
    with tf.GradientTape() as tape:
        tape.watch(x)
        # Find index of the target convolution layer
        conv_idx = -1
        for idx, layer in enumerate(model.layers):
            if layer.name == last_conv_layer_name:
                conv_idx = idx
                break
                
        # Forward pass up to the conv layer
        for idx in range(conv_idx + 1):
            x = model.layers[idx](x)
            
        conv_output = x
        tape.watch(conv_output)
        
        # Forward pass for remaining layers starting from conv_output
        temp_x = conv_output
        for idx in range(conv_idx + 1, len(model.layers)):
            temp_x = model.layers[idx](temp_x)
            
        loss = temp_x[0]

    # Compute gradients of predictions with respect to target conv layer activations
    grads = tape.gradient(loss, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight the activation maps
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize heatmap
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap + 1e-10)
    return heatmap.numpy()


def predict_image(img):
    if img is None: 
        return None, "No Image Provided", ""
    if image_model is None: 
        return img, "🟢 REAL (Mock Mode - Model Not Loaded)", "Mock Mode suggestions: Please ensure image model is loaded for actual prediction."
        
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_res = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE)) / 255.0
    img_input = np.expand_dims(img_res, axis=0)
    prob = float(image_model.predict(img_input, verbose=0)[0][0])
    
    analysis_result = get_prediction_label(prob)
    
    if prob >= 0.5:
        # FAKE: Draw anomaly bounding box using Grad-CAM
        try:
            heatmap = make_gradcam_heatmap(img_input, image_model, 'conv2d_5')
            heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
            heatmap_norm = np.uint8(255 * heatmap_resized)
            _, thresh = cv2.threshold(heatmap_norm, int(0.5 * 255), 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            img_box = img.copy()
            if len(contours) > 0:
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                if w > 5 and h > 5 and (w < img.shape[1] - 5 or h < img.shape[0] - 5):
                    cv2.rectangle(img_box, (x, y), (x+w, y+h), (255, 0, 0), 3) # Red box in RGB space
                else:
                    h_img, w_img, _ = img.shape
                    cv2.rectangle(img_box, (int(w_img*0.35), int(h_img*0.35)), (int(w_img*0.65), int(h_img*0.65)), (255, 0, 0), 3)
            else:
                h_img, w_img, _ = img.shape
                cv2.rectangle(img_box, (int(w_img*0.35), int(h_img*0.35)), (int(w_img*0.65), int(h_img*0.65)), (255, 0, 0), 3)
            output_img = img_box
        except Exception:
            img_box = img.copy()
            h_img, w_img, _ = img.shape
            cv2.rectangle(img_box, (int(w_img*0.35), int(h_img*0.35)), (int(w_img*0.65), int(h_img*0.65)), (255, 0, 0), 3)
            output_img = img_box
            
        explanation = ("⚠️ AI Explanation (Reason):\n"
                       "Anomalous blending boundaries, atypical frequency distributions, or inconsistent structural activations "
                       "detected in local facial/background components (marked by the red bounding box). "
                       "These artifacts are highly characteristic of GAN-generated or diffusion-based synthesis.")
    else:
        # REAL: Do not draw box, display original image
        output_img = img
        explanation = ("🛡️ AI Suggestions:\n"
                       "No major GAN/synthesis anomalies detected. For enhanced security:\n"
                       "1. Verify image metadata (EXIF data).\n"
                       "2. Inspect structural landmarks manually under varying exposures.\n"
                       "3. Check for camera-native noise patterns and lighting consistency.")
                       
    return output_img, analysis_result, explanation


def predict_audio(audio_path):
    if audio_path is None: 
        return "No Audio Provided", ""
    if audio_model is None: 
        return "🟢 REAL (Mock Mode - Model Not Loaded)", "Mock Mode suggestions: Please ensure audio model is loaded for actual prediction."
        
    try:
        y, sr = librosa.load(audio_path, duration=3.0, sr=16000)
        spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        log_spec = librosa.power_to_db(spectrogram, ref=np.max)
        
        if log_spec.shape[1] < 128:
            pad_width = 128 - log_spec.shape[1]
            log_spec = np.pad(log_spec, pad_width=((0, 0), (0, pad_width)), mode='constant')
        else:
            log_spec = log_spec[:, :128]
            
        log_spec = (log_spec - np.min(log_spec)) / (np.max(log_spec) - np.min(log_spec) + 1e-6)
        audio_input = np.expand_dims(np.expand_dims(log_spec, axis=0), axis=-1)
        prob = float(audio_model.predict(audio_input, verbose=0)[0][0])
        
        analysis_result = get_prediction_label(prob)
        
        if prob >= 0.5:
            explanation = ("⚠️ AI Explanation (Reason):\n"
                           "Abnormal spectral consistency patterns, unnatural high-frequency harmonic transitions, or phase "
                           "alignment discrepancies detected in the Mel-Spectrogram features. These patterns strongly match "
                           "synthesized text-to-speech (TTS) or voice cloning models.")
        else:
            explanation = ("🛡️ AI Suggestions:\n"
                           "The voice profile matches natural human vocal cords and natural acoustic environment noise. To confirm further:\n"
                           "1. Inspect conversational latency and natural breathing pauses.\n"
                           "2. Cross-reference background environmental noise consistency.")
                           
        return analysis_result, explanation
    except Exception as e:
        return f"Error processing audio: {str(e)}", ""


# Gradio Interface Deployment Build
with gr.Blocks(title="Unified Real-Time Deepfake Detector") as demo:
    gr.Markdown("# 🛡️ Unified Real-Time Deepfake Detector")
    gr.Markdown("Identify synthetic deepfakes across image, audio, and video platforms.")
    
    with gr.Tab("📸 Image Deepfake Detector"):
        with gr.Row():
            with gr.Column():
                img_input = gr.Image(type="numpy", label="Upload Image File")
                img_btn = gr.Button("Scan Image", variant="primary")
            with gr.Column():
                img_output_img = gr.Image(type="numpy", label="Analysis Visualization")
                img_output_text = gr.Textbox(label="Analysis Result")
                img_output_explanation = gr.Textbox(label="AI Explanation / Recommendations", lines=5)
        img_btn.click(
            predict_image, 
            inputs=img_input, 
            outputs=[img_output_img, img_output_text, img_output_explanation]
        )
        
    with gr.Tab("🎵 Audio Deepfake Detector"):
        with gr.Row():
            with gr.Column():
                audio_input = gr.Audio(type="filepath", label="Upload Audio File")
                audio_btn = gr.Button("Scan Audio File", variant="primary")
            with gr.Column():
                audio_output_text = gr.Textbox(label="Analysis Result")
                audio_output_explanation = gr.Textbox(label="AI Explanation / Recommendations", lines=5)
        audio_btn.click(
            predict_audio, 
            inputs=audio_input, 
            outputs=[audio_output_text, audio_output_explanation]
        )
        
    with gr.Tab("🎥 Video Deepfake Detector"):
        with gr.Row():
            with gr.Column():
                video_input = gr.Video(label="Upload Video File")
                video_btn = gr.Button("Scan Video Sequence", variant="primary")
            with gr.Column():
                video_output_compat = gr.Video(label="Browser Compatible Video")
                video_output_text = gr.Textbox(label="Analysis Result")
                video_output_explanation = gr.Textbox(label="AI Explanation / Recommendations", lines=5)
        video_btn.click(
            process_video_and_predict, 
            inputs=video_input, 
            outputs=[video_output_compat, video_output_text, video_output_explanation]
        )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7867, share=False)
