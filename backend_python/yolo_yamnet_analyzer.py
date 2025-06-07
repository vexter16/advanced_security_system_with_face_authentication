# import threading
# import subprocess
# import time
# import pandas as pd
# import tensorflow as tf
# import tensorflow_hub as hub
# import numpy as np
# import csv
# import cv2
# from scipy.io import wavfile
# from ultralytics import YOLO
# from collections import deque # Not used in this refactor but kept if you expand
# import queue # Not used in this refactor for display, results returned directly

# # --- CONFIGURATION DEFAULTS (can be overridden) ---
# DEFAULT_YOLO_MODEL_PATH = "best.pt" # Path relative to main_app.py or absolute
# DEFAULT_WAV_OUTPUT_PATH = "audio_temp.wav" # Temporary audio file

# DEFAULT_YOLO_CONFIDENCE_THRESHOLD = 0.45
# DEFAULT_YOLO_IOU_THRESHOLD = 0.7
# DEFAULT_FRAME_CHANGE_THRESHOLD_PERCENT = 3

# CLASS_NAME_MAP = {
#     'civilian': 'civilian', 'soldier': 'soldier', 'weapon': 'weapon', 'fence': 'fence',
# }
# CLASS_CIVILIAN = CLASS_NAME_MAP.get('civilian', 'civilian')
# CLASS_SOLDIER = CLASS_NAME_MAP.get('soldier', 'soldier')
# CLASS_WEAPON = CLASS_NAME_MAP.get('weapon', 'weapon')
# CLASS_FENCE = CLASS_NAME_MAP.get('fence', 'fence') # Added for completeness

# WEAPON_PROXIMITY_THRESHOLD = 75 # pixels

# DEFAULT_YAMNET_TARGET_SOUNDS = {
#     "threat": ["Gunshot, gunfire", "Explosion", "Machine gun", "Artillery fire"],
#     "fence_tampering": ["Squeak", "Scrape", "Metal", "Breaking", "Crowbar", "Hammer","Tick","Scissors","Glass","Tick-tock","Coin(dropping)","Whip","Finger Snapping","Mechanisms"],
#     "ambient": ["Speech", "Vehicle"], 
#     "alert": ["Alarm", "Siren"]
# }
# DEFAULT_YAMNET_CONFIDENCE_THRESHOLD = 0.25
# SUSTAINED_EVENT_THRESHOLD_FRAMES = 10 # Frames for an event to be "sustained"

# # === HELPER FUNCTIONS ===
# def get_center_bbox(bbox):
#     x1, y1, x2, y2 = bbox
#     return int((x1 + x2) / 2), int((y1 + y2) / 2)

# def calculate_distance_bboxes(bbox1, bbox2):
#     c1, c2 = get_center_bbox(bbox1), get_center_bbox(bbox2)
#     return np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)

# # === AUDIO PROCESSING LOGIC ===
# def process_audio_with_yamnet(video_path_for_audio, wav_output_path_audio=DEFAULT_WAV_OUTPUT_PATH, target_sounds=None, confidence_threshold=DEFAULT_YAMNET_CONFIDENCE_THRESHOLD):
#     if target_sounds is None:
#         target_sounds = DEFAULT_YAMNET_TARGET_SOUNDS

#     local_event_evidence = {
#         "explosion_sound_detected": False,
#         "gunfire_sound_detected": False,
#         "fence_tampering_sound_detected": False,
#     }
#     yamnet_results = {"detected_sounds": [], "highest_confidence_sound": ("None", 0.0)}

#     print(f"🔊 [YAMNet] Extracting audio from: {video_path_for_audio} to {wav_output_path_audio}")
#     try:
#         # Ensure paths with spaces are quoted
#         cmd = f'ffmpeg -i "{video_path_for_audio}" -ac 1 -ar 16000 -sample_fmt s16 "{wav_output_path_audio}" -y -hide_banner -loglevel error'
#         subprocess.run(cmd, shell=True, check=True, timeout=60)
#     except Exception as e:
#         print(f"🔊 [YAMNet] ERROR: ffmpeg failed for {video_path_for_audio}: {e}")
#         return local_event_evidence, yamnet_results # Return default/empty results

#     print(f"🔊 [YAMNet] Running audio classification for {video_path_for_audio}...")
#     try:
#         model_audio = hub.load('https://tfhub.dev/google/yamnet/1')
#         class_map_path = model_audio.class_map_path().numpy()
#         class_names_yamnet = [row['display_name'] for row in csv.DictReader(tf.io.gfile.GFile(class_map_path))]
        
#         sample_rate, wav_data = wavfile.read(wav_output_path_audio)
#         waveform = wav_data / (tf.int16.max if wav_data.dtype == np.int16 else np.iinfo(wav_data.dtype).max)
#         if len(waveform.shape) > 1: waveform = np.mean(waveform, axis=1)

#         scores, _, _ = model_audio(waveform)
#         scores_np = scores.numpy()

#         detected_sounds_this_file = []
#         max_conf = 0.0
#         best_sound = "None"

#         for i, class_name in enumerate(class_names_yamnet):
#             max_score_for_class = np.max(scores_np[:, i])
#             if max_score_for_class >= confidence_threshold:
#                 detected_sounds_this_file.append((class_name, float(max_score_for_class)))
#                 if max_score_for_class > max_conf:
#                     max_conf, best_sound = max_score_for_class, class_name
                
#                 if class_name in target_sounds["threat"]:
#                     if "Explosion" in class_name or "Artillery" in class_name:
#                         local_event_evidence["explosion_sound_detected"] = True
#                     elif "Gunshot" in class_name or "Machine gun" in class_name:
#                         local_event_evidence["gunfire_sound_detected"] = True
#                 if class_name in target_sounds["fence_tampering"]:
#                     local_event_evidence["fence_tampering_sound_detected"] = True
        
#         yamnet_results["detected_sounds"] = sorted(detected_sounds_this_file, key=lambda x: x[1], reverse=True)
#         yamnet_results["highest_confidence_sound"] = (best_sound, float(max_conf))

#         if detected_sounds_this_file:
#             print(f"🔊 [YAMNet] Detected for {video_path_for_audio}: {yamnet_results['detected_sounds']}")
#         else:
#             print(f"🔊 [YAMNet] No relevant sounds detected above threshold for {video_path_for_audio}.")
#     except Exception as e:
#         print(f"🔊 [YAMNet] ERROR in classification for {video_path_for_audio}: {e}")
    
#     # Clean up temporary audio file
#     if os.path.exists(wav_output_path_audio):
#         try:
#             os.remove(wav_output_path_audio)
#         except Exception as e_rem:
#             print(f"🔊 [YAMNet] Warning: Could not remove temp audio file {wav_output_path_audio}: {e_rem}")
            
#     return local_event_evidence, yamnet_results

# # === PER-FRAME ANALYSIS (called by YOLO processor) ===
# def analyze_yolo_frame_for_events(frame_detections_data, yamnet_sound_evidence_for_video):
#     """Analyzes a single frame's YOLO detections in context of video-wide YAMNet sounds."""
#     per_frame_event_evidence = {
#         "is_civilian_with_weapon_this_frame": False,
#         "is_civilian_near_fence_with_sound_this_frame": False,
#         "is_unarmed_civilian_this_frame": False,
#         "is_soldier_activity_this_frame": False,
#         "primary_threat_type_this_frame": "other" # Default
#     }
#     detections = frame_detections_data.get('detections', [])

#     civilians = [d for d in detections if d['class_name'] == CLASS_CIVILIAN]
#     soldiers = [d for d in detections if d['class_name'] == CLASS_SOLDIER]
#     weapons = [d for d in detections if d['class_name'] == CLASS_WEAPON]
#     # fences = [d for d in detections if d['class_name'] == CLASS_FENCE] # If YOLO detects fences

#     # Civilian with Weapon
#     for civ in civilians:
#         for wep in weapons:
#             if calculate_distance_bboxes(civ['bbox'], wep['bbox']) < WEAPON_PROXIMITY_THRESHOLD:
#                 per_frame_event_evidence["is_civilian_with_weapon_this_frame"] = True
#                 per_frame_event_evidence["primary_threat_type_this_frame"] = "civilian_with_weapon"
#                 break
#         if per_frame_event_evidence["is_civilian_with_weapon_this_frame"]: break
    
#     # Civilian near Fence with Tampering Sound (from overall video audio)
#     if civilians and yamnet_sound_evidence_for_video.get("fence_tampering_sound_detected", False):
#         # Note: This check is simplistic. Real "near fence" logic would need fence detections from YOLO.
#         # For now, if tampering sound present for the video, any civilian frame contributes to this.
#         per_frame_event_evidence["is_civilian_near_fence_with_sound_this_frame"] = True
#         if not per_frame_event_evidence["is_civilian_with_weapon_this_frame"]: # Don't override if armed
#              per_frame_event_evidence["primary_threat_type_this_frame"] = "civilian_fence_tamper"

#     # Unarmed Civilian Presence
#     if civilians and not per_frame_event_evidence["is_civilian_with_weapon_this_frame"]:
#         per_frame_event_evidence["is_unarmed_civilian_this_frame"] = True
#         if per_frame_event_evidence["primary_threat_type_this_frame"] == "other": # Don't override higher threats
#              per_frame_event_evidence["primary_threat_type_this_frame"] = "unarmed_civilian"

#     # Soldier Activity
#     if soldiers and not civilians: # Simple assumption
#         per_frame_event_evidence["is_soldier_activity_this_frame"] = True
#         if per_frame_event_evidence["primary_threat_type_this_frame"] == "other":
#             per_frame_event_evidence["primary_threat_type_this_frame"] = "soldier_activity"
            
#     return per_frame_event_evidence

# # === VIDEO THREAD: YOLO Video Processing (refactored as a callable function) ===
# def process_video_with_yolo(video_path_yolo, yolo_model_path_yolo=DEFAULT_YOLO_MODEL_PATH, yamnet_sound_evidence_for_video=None):
#     if yamnet_sound_evidence_for_video is None:
#         yamnet_sound_evidence_for_video = {} # Default empty evidence

#     yolo_results_per_frame = [] # Detections for each processed frame
#     aggregated_event_counts = {
#         "civilian_with_weapon_frames": 0,
#         "civilian_near_fence_with_sound_frames": 0,
#         "unarmed_civilian_frames": 0,
#         "soldier_activity_frames": 0,
#         "first_significant_event_frame_id": -1,
#         "frames_for_qwen_raw": [] # Will store frame_bgr, timestamp, id, detections
#     }

#     print(f"🎥 [YOLO] Initializing YOLO model from: {yolo_model_path_yolo}")
#     try:
#         model_yolo = YOLO(yolo_model_path_yolo)
#         yolo_class_names_dict = model_yolo.names
#     except Exception as e:
#         print(f"🎥 [YOLO] ERROR: Could not load YOLO model: {e}")
#         return yolo_results_per_frame, aggregated_event_counts

#     print(f"🎥 [YOLO] Opening video file: {video_path_yolo}")
#     cap = cv2.VideoCapture(video_path_yolo)
#     if not cap.isOpened():
#         print(f"🎥 [YOLO] ERROR: Could not open video file: {video_path_yolo}")
#         return yolo_results_per_frame, aggregated_event_counts

#     frame_id_counter = 0
#     fps = cap.get(cv2.CAP_PROP_FPS)
#     if fps == 0: fps = 25.0  # Default FPS if not readable

#     prev_gray_frame_eq = None
    
#     print(f"🎥 [YOLO] Starting video processing for {video_path_yolo}...")
#     while cap.isOpened():
#         success, frame_bgr = cap.read()
#         if not success:
#             break

#         gray_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
#         gray_frame_eq = cv2.equalizeHist(gray_frame)
#         if prev_gray_frame_eq is not None:
#             frame_diff = cv2.absdiff(gray_frame_eq, prev_gray_frame_eq)
#             if frame_diff.size > 0:
#                 diff_score = np.sum(frame_diff) / (frame_diff.size * 255) * 100
#                 if diff_score < DEFAULT_FRAME_CHANGE_THRESHOLD_PERCENT:
#                     frame_id_counter += 1
#                     prev_gray_frame_eq = gray_frame_eq.copy()
#                     continue # Skip this frame
#         prev_gray_frame_eq = gray_frame_eq.copy()

#         yolo_model_outputs = model_yolo.predict(frame_bgr, save=False, iou=DEFAULT_YOLO_IOU_THRESHOLD, conf=DEFAULT_YOLO_CONFIDENCE_THRESHOLD, verbose=False)
        
#         timestamp_sec = frame_id_counter / fps
#         time_str = time.strftime('%H:%M:%S', time.gmtime(timestamp_sec)) + f".{int((timestamp_sec % 1) * 1000):03d}"
        
#         current_frame_detections_list = []
#         for result in yolo_model_outputs:
#             for box in result.boxes:
#                 x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
#                 class_id = int(box.cls[0].item())
#                 prob = round(box.conf[0].item(), 2)
#                 class_name_detected = yolo_class_names_dict.get(class_id, "unknown")
                
#                 current_frame_detections_list.append({
#                     'bbox': [x1, y1, x2, y2], 'class_id': class_id,
#                     'class_name': class_name_detected, 'confidence': prob
#                 })
        
#         frame_data_for_analysis = {
#             'frame_id': frame_id_counter,
#             'timestamp': time_str,
#             'detections': current_frame_detections_list
#             # Not storing full frame here to save memory, Qwen frames collected later if needed
#         }
#         yolo_results_per_frame.append(frame_data_for_analysis)
        
#         # Store raw frame data for potential Qwen submission
#         aggregated_event_counts["frames_for_qwen_raw"].append({
#             'frame_id': frame_id_counter,
#             'timestamp': time_str,
#             'frame_bgr': frame_bgr, # Store the actual frame
#             'detections': current_frame_detections_list
#         })


#         if current_frame_detections_list:
#             frame_event_info = analyze_yolo_frame_for_events(frame_data_for_analysis, yamnet_sound_evidence_for_video)
#             if frame_event_info["is_civilian_with_weapon_this_frame"]:
#                 aggregated_event_counts["civilian_with_weapon_frames"] += 1
#             if frame_event_info["is_civilian_near_fence_with_sound_this_frame"]:
#                 aggregated_event_counts["civilian_near_fence_with_sound_frames"] += 1
#             if frame_event_info["is_unarmed_civilian_this_frame"]:
#                 aggregated_event_counts["unarmed_civilian_frames"] += 1
#             if frame_event_info["is_soldier_activity_this_frame"]:
#                 aggregated_event_counts["soldier_activity_frames"] += 1
            
#             if aggregated_event_counts["first_significant_event_frame_id"] == -1:
#                 is_significant_sound = yamnet_sound_evidence_for_video.get("explosion_sound_detected", False) or \
#                                        yamnet_sound_evidence_for_video.get("gunfire_sound_detected", False)
#                 if frame_event_info["primary_threat_type_this_frame"] in ["civilian_with_weapon", "civilian_fence_tamper"] or is_significant_sound:
#                     aggregated_event_counts["first_significant_event_frame_id"] = frame_id_counter
        
#         frame_id_counter += 1
#         if frame_id_counter % 100 == 0:
#             print(f"🎥 [YOLO] Processed frame {frame_id_counter} for {video_path_yolo}")

#     cap.release()
#     print(f"🎥 [YOLO] Finished video processing for {video_path_yolo}.")
#     return yolo_results_per_frame, aggregated_event_counts


# def determine_primary_event_and_qwen_package(yamnet_audio_evidence, yolo_aggregated_evidence, all_raw_frames_data):
#     primary_event_type = "No Significant Event"
#     qwen_prompt = "Perform a general scene analysis."
#     frames_for_qwen_output = [] # List of dictionaries with 'frame_id', 'timestamp', 'frame_image_bytes', 'detections'

#     start_frame_id_for_qwen = yolo_aggregated_evidence.get("first_significant_event_frame_id", -1)

#     # Prioritize based on combined evidence
#     if yamnet_audio_evidence.get("explosion_sound_detected"):
#         primary_event_type = "Explosion Event"
#         qwen_prompt = "CRITICAL ALERT: Explosion sound detected. Analyze subsequent video frames for visual confirmation, damage, casualties, and ongoing threats."
#         if start_frame_id_for_qwen == -1 : start_frame_id_for_qwen = 0 # Default to start if not set by YOLO
#     elif yamnet_audio_evidence.get("gunfire_sound_detected"):
#         primary_event_type = "Gunfire Event"
#         qwen_prompt = "CRITICAL ALERT: Gunfire sound detected. Analyze subsequent video frames for shooters, targets, casualties, and tactical situation."
#         if start_frame_id_for_qwen == -1 : start_frame_id_for_qwen = 0
#     elif yolo_aggregated_evidence.get("civilian_with_weapon_frames", 0) >= SUSTAINED_EVENT_THRESHOLD_FRAMES:
#         primary_event_type = "Civilian with Weapon"
#         qwen_prompt = f"POTENTIAL THREAT: Civilian observed with a weapon for a sustained period ({yolo_aggregated_evidence['civilian_with_weapon_frames']} frames). Analyze actions, weapon type, intent, and surrounding context from the point of first detection onwards."
#         # start_frame_id_for_qwen should already be set by YOLO logic
#     elif yolo_aggregated_evidence.get("civilian_near_fence_with_sound_frames", 0) >= SUSTAINED_EVENT_THRESHOLD_FRAMES and \
#          yamnet_audio_evidence.get("fence_tampering_sound_detected"):
#         primary_event_type = "Fence Breach Attempt"
#         qwen_prompt = f"SECURITY ALERT: Civilian activity near fence detected for {yolo_aggregated_evidence['civilian_near_fence_with_sound_frames']} frames, with concurrent audio suggesting fence tampering. Analyze interaction with the fence and assess breach attempt from the point of first detection onwards."
#     elif yolo_aggregated_evidence.get("unarmed_civilian_frames", 0) >= SUSTAINED_EVENT_THRESHOLD_FRAMES * 2:
#         primary_event_type = "Civilian Presence (Unarmed)"
#         qwen_prompt = f"INFO: Sustained civilian presence observed ({yolo_aggregated_evidence['unarmed_civilian_frames']} frames) without obvious weapons. Monitor activity, assess intentions, and identify any unusual behavior."
#         if start_frame_id_for_qwen == -1 and yolo_aggregated_evidence["unarmed_civilian_frames"] > 0:
#             # Find first unarmed civilian frame if not set by a more "significant" event
#             for frame_meta in all_raw_frames_data:
#                 if any(d['class_name'] == CLASS_CIVILIAN for d in frame_meta['detections']):
#                     is_armed_in_this_specific_frame = False
#                     for civ_d in (d for d in frame_meta['detections'] if d['class_name'] == CLASS_CIVILIAN):
#                         for wep_d in (d for d in frame_meta['detections'] if d['class_name'] == CLASS_WEAPON):
#                             if calculate_distance_bboxes(civ_d['bbox'], wep_d['bbox']) < WEAPON_PROXIMITY_THRESHOLD:
#                                 is_armed_in_this_specific_frame = True; break
#                         if is_armed_in_this_specific_frame: break
#                     if not is_armed_in_this_specific_frame:
#                         start_frame_id_for_qwen = frame_meta['frame_id']; break
#             if start_frame_id_for_qwen == -1: start_frame_id_for_qwen = 0 # fallback if still not found
#     elif yolo_aggregated_evidence.get("soldier_activity_frames", 0) > 0:
#         primary_event_type = "Routine Soldier Activity"
#         qwen_prompt = "INFO: Predominantly soldier activity observed. Provide a general summary of the activities."
#         start_frame_id_for_qwen = 0 
#     else:
#         primary_event_type = "No Specific Classified Event"
#         qwen_prompt = "The video has been processed. No specific pre-defined critical events were detected based on current rules. Perform a general analysis if desired."
#         start_frame_id_for_qwen = 0

#     # Collect frames for Qwen based on start_frame_id_for_qwen
#     if start_frame_id_for_qwen != -1:
#         print(f"Primary event '{primary_event_type}' considered to start around frame {start_frame_id_for_qwen}. Collecting frames for Qwen.")
#         for frame_meta in all_raw_frames_data: # Iterate through the raw frames stored by YOLO
#             if frame_meta['frame_id'] >= start_frame_id_for_qwen:
#                 if frame_meta['frame_bgr'] is not None:
#                     # Encode frame to JPEG bytes for Qwen or API transmission
#                     success, encoded_image = cv2.imencode('.jpg', frame_meta['frame_bgr'])
#                     if success:
#                         frames_for_qwen_output.append({
#                             'frame_id': frame_meta['frame_id'],
#                             'timestamp': frame_meta['timestamp'],
#                             'frame_image_base64': base64.b64encode(encoded_image).decode('utf-8'), # Send as base64
#                             'detections': frame_meta['detections'] # Detections for this specific frame
#                         })
#                     else:
#                         print(f"Warning: Failed to encode frame {frame_meta['frame_id']} for Qwen.")
        
#         # Limit number of frames for Qwen if too many (e.g., max 50-100 frames or ~5 seconds)
#         MAX_QWEN_FRAMES = 50 
#         if len(frames_for_qwen_output) > MAX_QWEN_FRAMES:
#             print(f"Trimming Qwen frames from {len(frames_for_qwen_output)} to {MAX_QWEN_FRAMES}")
#             # Simple trim from the start, or implement smarter keyframe selection
#             frames_for_qwen_output = frames_for_qwen_output[:MAX_QWEN_FRAMES]


#     if not frames_for_qwen_output and primary_event_type not in ["No Significant Event", "No Specific Classified Event", "Routine Soldier Activity"]:
#         print(f"Warning: Event '{primary_event_type}' determined, but no frames were collected for Qwen based on start_frame_id.")
#     elif not frames_for_qwen_output and primary_event_type == "Routine Soldier Activity":
#         # For routine, if no specific start, maybe send first few with detections
#         count = 0
#         for frame_meta in all_raw_frames_data:
#             if frame_meta['detections']: # Only send frames that had some YOLO detections
#                 success, encoded_image = cv2.imencode('.jpg', frame_meta['frame_bgr'])
#                 if success:
#                     frames_for_qwen_output.append({
#                         'frame_id': frame_meta['frame_id'], 'timestamp': frame_meta['timestamp'],
#                         'frame_image_base64': base64.b64encode(encoded_image).decode('utf-8'),
#                         'detections': frame_meta['detections']
#                     })
#                     count += 1
#                     if count >= 5: break # Send up to 5 sample frames for routine
#         if frames_for_qwen_output:
#             print(f"For '{primary_event_type}', sending {len(frames_for_qwen_output)} sample frames to Qwen.")


#     return primary_event_type, qwen_prompt, frames_for_qwen_output

# # Main analysis function to be called by Flask
# def analyze_video_full(video_path_main):
#     # Reset global state for each new video analysis
#     global_event_evidence_for_video = {
#         "explosion_sound_detected": False, "gunfire_sound_detected": False,
#         "fence_tampering_sound_detected": False,
#     } # This is specific to audio only. YOLO aggregates its own counts.

#     yamnet_local_evidence, yamnet_overall_results_for_video = process_audio_with_yamnet(video_path_main)
    
#     # Pass the YAMNet evidence to the YOLO processor so it can use it for per-frame analysis context
#     yolo_results_per_frame, yolo_aggregated_counts = process_video_with_yolo(video_path_main, yamnet_sound_evidence_for_video=yamnet_local_evidence)
    
#     # Get the raw frames with detections that YOLO stored
#     # (this was stored in `aggregated_event_counts["frames_for_qwen_raw"]` inside process_video_with_yolo)
#     # For clarity, let's assume `yolo_aggregated_counts` contains `frames_for_qwen_raw`
#     all_raw_frames_data_from_yolo = yolo_aggregated_counts.pop("frames_for_qwen_raw", [])


#     primary_event, qwen_prompt, qwen_frames_package = determine_primary_event_and_qwen_package(
#         yamnet_local_evidence, # Audio evidence
#         yolo_aggregated_counts,  # YOLO frame counts
#         all_raw_frames_data_from_yolo # List of dicts with frame_bgr, id, timestamp, detections
#     )

#     analysis_summary = {
#         "video_path": video_path_main,
#         "yamnet_results": yamnet_overall_results_for_video,
#         "yolo_detections_summary": yolo_aggregated_counts, # Frame counts for event types
#         "yolo_per_frame_details_count": len(yolo_results_per_frame), # How many frames had any yolo detection
#         "primary_event_determined": primary_event,
#         "qwen_submission_plan": {
#             "prompt": qwen_prompt,
#             "frames_count": len(qwen_frames_package),
#             # Do not send full frame data in this summary, only count.
#             # Actual frames are in qwen_frames_package (which will be sent to Qwen separately if needed)
#         }
#     }
#     # For now, the function calling Qwen will use `qwen_prompt` and `qwen_frames_package`
#     return analysis_summary, qwen_prompt, qwen_frames_package

# if __name__ == '__main__':
#     # This is for local testing of this module
#     test_video_path = '/Users/veeshal/Downloads/Video_Request_Fence_Approach.mp4' # Replace with a test video
#     if os.path.exists(test_video_path):
#         summary, q_prompt, q_frames = analyze_video_full(test_video_path)
#         print("\n--- ANALYSIS SUMMARY ---")
#         import json
#         print(json.dumps(summary, indent=2))
#         print(f"\nQwen Prompt: {q_prompt}")
#         print(f"Frames prepared for Qwen: {len(q_frames)}")
#         if q_frames:
#             print(f"First Qwen frame ID: {q_frames[0]['frame_id']}, Detections: {len(q_frames[0]['detections'])}")
#             # For testing, you could save one of the base64 images to check
#             # test_img_data = base64.b64decode(q_frames[0]['frame_image_base64'])
#             # with open("test_qwen_frame.jpg", "wb") as f:
#             #     f.write(test_img_data)
#             # print("Saved test_qwen_frame.jpg")

#     else:
#         print(f"Test video not found: {test_video_path}")

import threading
import subprocess
import time
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import csv
import cv2
from scipy.io import wavfile
from ultralytics import YOLO
# from collections import deque # Not strictly needed for this version of logic
# import queue # Not strictly needed for this version of logic
import os # For os.path.exists
import base64 # For encoding frames for Qwen

# === CONFIGURATION DEFAULTS (can be overridden) ===
DEFAULT_YOLO_MODEL_PATH = "best.pt" # Default if no override is given
DEFAULT_WAV_OUTPUT_PATH = "audio_temp.wav"

DEFAULT_YOLO_CONFIDENCE_THRESHOLD = 0.45
DEFAULT_YOLO_IOU_THRESHOLD = 0.7
DEFAULT_FRAME_CHANGE_THRESHOLD_PERCENT = 20

CLASS_NAME_MAP = {
    'civilian': 'civilian', 'soldier': 'soldier', 'weapon': 'weapon', 'fence': 'fence',
}
CLASS_CIVILIAN = CLASS_NAME_MAP.get('civilian', 'civilian')
CLASS_SOLDIER = CLASS_NAME_MAP.get('soldier', 'soldier')
CLASS_WEAPON = CLASS_NAME_MAP.get('weapon', 'weapon')
CLASS_FENCE = CLASS_NAME_MAP.get('fence', 'fence')

WEAPON_PROXIMITY_THRESHOLD = 75

DEFAULT_YAMNET_TARGET_SOUNDS = {
    "threat": ["Gunshot, gunfire", "Explosion", "Machine gun", "Artillery fire"],
    "fence_tampering": ["Squeak", "Scrape", "Metal", "Breaking", "Crowbar", "Hammer","Tick","Scissors","Glass","Tick-tock","Coin(dropping)","Whip","Finger Snapping","Mechanisms"],
    "ambient": ["Speech", "Vehicle"], 
    "alert": ["Alarm", "Siren"]
}
DEFAULT_YAMNET_CONFIDENCE_THRESHOLD = 0.25
SUSTAINED_EVENT_THRESHOLD_FRAMES = 10

# === HELPER FUNCTIONS ===
def get_center_bbox(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)

def calculate_distance_bboxes(bbox1, bbox2):
    c1, c2 = get_center_bbox(bbox1), get_center_bbox(bbox2)
    return np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)

# === AUDIO PROCESSING LOGIC (Should be mostly fine) ===
def process_audio_with_yamnet(video_path_for_audio, wav_output_path_audio=DEFAULT_WAV_OUTPUT_PATH, target_sounds=None, confidence_threshold=DEFAULT_YAMNET_CONFIDENCE_THRESHOLD):
    if target_sounds is None:
        target_sounds = DEFAULT_YAMNET_TARGET_SOUNDS

    local_event_evidence = {
        "explosion_sound_detected": False,
        "gunfire_sound_detected": False,
        "fence_tampering_sound_detected": False,
    }
    yamnet_results = {"detected_sounds": [], "highest_confidence_sound": ("None", 0.0)}

    print(f"🔊 [YAMNet] Extracting audio from: {video_path_for_audio} to {wav_output_path_audio}")
    try:
        cmd = f'ffmpeg -i "{video_path_for_audio}" -ac 1 -ar 16000 -sample_fmt s16 "{wav_output_path_audio}" -y -hide_banner -loglevel error'
        subprocess.run(cmd, shell=True, check=True, timeout=60)
    except Exception as e:
        print(f"🔊 [YAMNet] ERROR: ffmpeg failed for {video_path_for_audio}: {e}")
        return local_event_evidence, yamnet_results

    print(f"🔊 [YAMNet] Running audio classification for {video_path_for_audio}...")
    try:
        model_audio = hub.load('https://tfhub.dev/google/yamnet/1')
        class_map_path = model_audio.class_map_path().numpy()
        class_names_yamnet = [row['display_name'] for row in csv.DictReader(tf.io.gfile.GFile(class_map_path))]
        
        sample_rate, wav_data = wavfile.read(wav_output_path_audio)
        waveform = wav_data / (tf.int16.max if wav_data.dtype == np.int16 else np.iinfo(wav_data.dtype).max)
        if len(waveform.shape) > 1: waveform = np.mean(waveform, axis=1)

        scores, _, _ = model_audio(waveform)
        scores_np = scores.numpy()

        detected_sounds_this_file = []
        max_conf = 0.0
        best_sound = "None"

        for i, class_name in enumerate(class_names_yamnet):
            max_score_for_class = np.max(scores_np[:, i])
            if max_score_for_class >= confidence_threshold:
                detected_sounds_this_file.append((class_name, float(max_score_for_class)))
                if max_score_for_class > max_conf:
                    max_conf, best_sound = max_score_for_class, class_name
                
                if class_name in target_sounds["threat"]:
                    if "Explosion" in class_name or "Artillery" in class_name:
                        local_event_evidence["explosion_sound_detected"] = True
                    elif "Gunshot" in class_name or "Machine gun" in class_name:
                        local_event_evidence["gunfire_sound_detected"] = True
                if class_name in target_sounds["fence_tampering"]:
                    local_event_evidence["fence_tampering_sound_detected"] = True
        
        yamnet_results["detected_sounds"] = sorted(detected_sounds_this_file, key=lambda x: x[1], reverse=True)
        yamnet_results["highest_confidence_sound"] = (best_sound, float(max_conf))

        if detected_sounds_this_file:
            print(f"🔊 [YAMNet] Detected for {video_path_for_audio}: {yamnet_results['detected_sounds']}")
        else:
            print(f"🔊 [YAMNet] No relevant sounds detected above threshold for {video_path_for_audio}.")
    except Exception as e:
        print(f"🔊 [YAMNet] ERROR in classification for {video_path_for_audio}: {e}")
    
    if os.path.exists(wav_output_path_audio):
        try: os.remove(wav_output_path_audio)
        except Exception as e_rem: print(f"🔊 [YAMNet] Warning: Could not remove temp audio file {wav_output_path_audio}: {e_rem}")
            
    return local_event_evidence, yamnet_results

# === PER-FRAME ANALYSIS (Should be fine) ===
def analyze_yolo_frame_for_events(frame_detections_data, yamnet_sound_evidence_for_video):
    per_frame_event_evidence = {
        "is_civilian_with_weapon_this_frame": False,
        "is_civilian_near_fence_with_sound_this_frame": False,
        "is_unarmed_civilian_this_frame": False,
        "is_soldier_activity_this_frame": False,
        "primary_threat_type_this_frame": "other"
    }
    detections = frame_detections_data.get('detections', [])
    civilians = [d for d in detections if d['class_name'] == CLASS_CIVILIAN]
    soldiers = [d for d in detections if d['class_name'] == CLASS_SOLDIER]
    weapons = [d for d in detections if d['class_name'] == CLASS_WEAPON]

    for civ in civilians:
        for wep in weapons:
            if calculate_distance_bboxes(civ['bbox'], wep['bbox']) < WEAPON_PROXIMITY_THRESHOLD:
                per_frame_event_evidence["is_civilian_with_weapon_this_frame"] = True
                per_frame_event_evidence["primary_threat_type_this_frame"] = "civilian_with_weapon"
                break
        if per_frame_event_evidence["is_civilian_with_weapon_this_frame"]: break
    
    if civilians and yamnet_sound_evidence_for_video.get("fence_tampering_sound_detected", False):
        per_frame_event_evidence["is_civilian_near_fence_with_sound_this_frame"] = True
        if not per_frame_event_evidence["is_civilian_with_weapon_this_frame"]:
             per_frame_event_evidence["primary_threat_type_this_frame"] = "civilian_fence_tamper"

    if civilians and not per_frame_event_evidence["is_civilian_with_weapon_this_frame"]:
        per_frame_event_evidence["is_unarmed_civilian_this_frame"] = True
        if per_frame_event_evidence["primary_threat_type_this_frame"] == "other":
             per_frame_event_evidence["primary_threat_type_this_frame"] = "unarmed_civilian"

    if soldiers and not civilians: 
        per_frame_event_evidence["is_soldier_activity_this_frame"] = True
        if per_frame_event_evidence["primary_threat_type_this_frame"] == "other":
            per_frame_event_evidence["primary_threat_type_this_frame"] = "soldier_activity"
            
    return per_frame_event_evidence

# === VIDEO THREAD: YOLO Video Processing ===
# *** MODIFIED FUNCTION SIGNATURE TO ACCEPT yolo_model_path_yolo ***
def process_video_with_yolo(video_path_yolo, yolo_model_path_yolo=DEFAULT_YOLO_MODEL_PATH, yamnet_sound_evidence_for_video=None):
    if yamnet_sound_evidence_for_video is None:
        yamnet_sound_evidence_for_video = {}

    yolo_results_per_frame = []
    aggregated_event_counts = {
        "civilian_with_weapon_frames": 0,
        "civilian_near_fence_with_sound_frames": 0,
        "unarmed_civilian_frames": 0,
        "soldier_activity_frames": 0,
        "first_significant_event_frame_id": -1,
        "frames_for_qwen_raw": []
    }

    print(f"🎥 [YOLO] Initializing YOLO model from: {yolo_model_path_yolo}") # Uses the passed path
    try:
        model_yolo = YOLO(yolo_model_path_yolo) # *** USES THE PASSED PATH ***
        yolo_class_names_dict = model_yolo.names
    except Exception as e:
        print(f"🎥 [YOLO] ERROR: Could not load YOLO model ({yolo_model_path_yolo}): {e}")
        return yolo_results_per_frame, aggregated_event_counts

    # ... (rest of the process_video_with_yolo function remains the same as your original version) ...
    # Ensure all logic for frame reading, processing, and appending to results is here.
    # For brevity, I'm not pasting the entire loop again, but it should be identical to what you provided.
    # The key is that `model_yolo = YOLO(yolo_model_path_yolo)` uses the correct path.

    print(f"🎥 [YOLO] Opening video file: {video_path_yolo}")
    cap = cv2.VideoCapture(video_path_yolo)
    if not cap.isOpened():
        print(f"🎥 [YOLO] ERROR: Could not open video file: {video_path_yolo}")
        return yolo_results_per_frame, aggregated_event_counts

    frame_id_counter = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 25.0  

    prev_gray_frame_eq = None
    
    print(f"🎥 [YOLO] Starting video processing for {video_path_yolo}...")
    while cap.isOpened():
        success, frame_bgr = cap.read()
        if not success:
            break

        gray_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray_frame_eq = cv2.equalizeHist(gray_frame)
        if prev_gray_frame_eq is not None:
            frame_diff = cv2.absdiff(gray_frame_eq, prev_gray_frame_eq)
            if frame_diff.size > 0:
                diff_score = np.sum(frame_diff) / (frame_diff.size * 255) * 100
                if diff_score < DEFAULT_FRAME_CHANGE_THRESHOLD_PERCENT:
                    frame_id_counter += 1
                    prev_gray_frame_eq = gray_frame_eq.copy()
                    continue 
        prev_gray_frame_eq = gray_frame_eq.copy()

        yolo_model_outputs = model_yolo.predict(frame_bgr, save=False, iou=DEFAULT_YOLO_IOU_THRESHOLD, conf=DEFAULT_YOLO_CONFIDENCE_THRESHOLD, verbose=False)
        
        timestamp_sec = frame_id_counter / fps
        time_str = time.strftime('%H:%M:%S', time.gmtime(timestamp_sec)) + f".{int((timestamp_sec % 1) * 1000):03d}"
        
        current_frame_detections_list = []
        for result in yolo_model_outputs:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                class_id = int(box.cls[0].item())
                prob = round(box.conf[0].item(), 2)
                class_name_detected = yolo_class_names_dict.get(class_id, "unknown")
                
                current_frame_detections_list.append({
                    'bbox': [x1, y1, x2, y2], 'class_id': class_id,
                    'class_name': class_name_detected, 'confidence': prob
                })
        
        frame_data_for_analysis = {
            'frame_id': frame_id_counter,
            'timestamp': time_str,
            'detections': current_frame_detections_list
        }
        yolo_results_per_frame.append(frame_data_for_analysis)
        
        aggregated_event_counts["frames_for_qwen_raw"].append({
            'frame_id': frame_id_counter,
            'timestamp': time_str,
            'frame_bgr': frame_bgr, 
            'detections': current_frame_detections_list
        })

        if current_frame_detections_list:
            frame_event_info = analyze_yolo_frame_for_events(frame_data_for_analysis, yamnet_sound_evidence_for_video)
            if frame_event_info["is_civilian_with_weapon_this_frame"]:
                aggregated_event_counts["civilian_with_weapon_frames"] += 1
            if frame_event_info["is_civilian_near_fence_with_sound_this_frame"]:
                aggregated_event_counts["civilian_near_fence_with_sound_frames"] += 1
            if frame_event_info["is_unarmed_civilian_this_frame"]:
                aggregated_event_counts["unarmed_civilian_frames"] += 1
            if frame_event_info["is_soldier_activity_this_frame"]:
                aggregated_event_counts["soldier_activity_frames"] += 1
            
            if aggregated_event_counts["first_significant_event_frame_id"] == -1:
                is_significant_sound = yamnet_sound_evidence_for_video.get("explosion_sound_detected", False) or \
                                       yamnet_sound_evidence_for_video.get("gunfire_sound_detected", False)
                if frame_event_info["primary_threat_type_this_frame"] in ["civilian_with_weapon", "civilian_fence_tamper"] or is_significant_sound:
                    aggregated_event_counts["first_significant_event_frame_id"] = frame_id_counter
        
        frame_id_counter += 1
        if frame_id_counter % 100 == 0:
            print(f"🎥 [YOLO] Processed frame {frame_id_counter} for {video_path_yolo}")

    cap.release()
    print(f"🎥 [YOLO] Finished video processing for {video_path_yolo}.")
    return yolo_results_per_frame, aggregated_event_counts


# === FUNCTION TO DETERMINE PRIMARY EVENT AND PREPARE QWEN DATA (Should be fine) ===
def determine_primary_event_and_qwen_package(yamnet_audio_evidence, yolo_aggregated_evidence, all_raw_frames_data):
    primary_event_type = "No Significant Event"
    qwen_prompt = "Perform a general scene analysis."
    frames_for_qwen_output = [] 

    start_frame_id_for_qwen = yolo_aggregated_evidence.get("first_significant_event_frame_id", -1)

    if yamnet_audio_evidence.get("explosion_sound_detected"):
        primary_event_type = "Explosion Event"
        qwen_prompt = "CRITICAL ALERT: Explosion sound detected. Analyze subsequent video frames for visual confirmation, damage, casualties, and ongoing threats."
        if start_frame_id_for_qwen == -1 : start_frame_id_for_qwen = 0 
    elif yamnet_audio_evidence.get("gunfire_sound_detected"):
        primary_event_type = "Gunfire Event"
        qwen_prompt = "CRITICAL ALERT: Gunfire sound detected. Analyze subsequent video frames for shooters, targets, casualties, and tactical situation."
        if start_frame_id_for_qwen == -1 : start_frame_id_for_qwen = 0
    elif yolo_aggregated_evidence.get("civilian_with_weapon_frames", 0) >= SUSTAINED_EVENT_THRESHOLD_FRAMES:
        primary_event_type = "Civilian with Weapon"
        qwen_prompt = f"POTENTIAL THREAT: Civilian observed with a weapon for a sustained period ({yolo_aggregated_evidence['civilian_with_weapon_frames']} frames). Analyze actions, weapon type, intent, and surrounding context from the point of first detection onwards."
    elif yolo_aggregated_evidence.get("civilian_near_fence_with_sound_frames", 0) >= SUSTAINED_EVENT_THRESHOLD_FRAMES and \
         yamnet_audio_evidence.get("fence_tampering_sound_detected"):
        primary_event_type = "Fence Breach Attempt"
        qwen_prompt = f"SECURITY ALERT: Civilian activity near fence detected for {yolo_aggregated_evidence['civilian_near_fence_with_sound_frames']} frames, with concurrent audio suggesting fence tampering. Analyze interaction with the fence and assess breach attempt from the point of first detection onwards."
    elif yolo_aggregated_evidence.get("unarmed_civilian_frames", 0) >= SUSTAINED_EVENT_THRESHOLD_FRAMES * 2:
        primary_event_type = "Civilian Presence (Unarmed)"
        qwen_prompt = f"INFO: Sustained civilian presence observed ({yolo_aggregated_evidence['unarmed_civilian_frames']} frames) without obvious weapons. Monitor activity, assess intentions, and identify any unusual behavior."
        if start_frame_id_for_qwen == -1 and yolo_aggregated_evidence["unarmed_civilian_frames"] > 0:
            for frame_meta in all_raw_frames_data:
                if any(d['class_name'] == CLASS_CIVILIAN for d in frame_meta['detections']):
                    is_armed_in_this_specific_frame = False
                    for civ_d in (d for d in frame_meta['detections'] if d['class_name'] == CLASS_CIVILIAN):
                        for wep_d in (d for d in frame_meta['detections'] if d['class_name'] == CLASS_WEAPON):
                            if calculate_distance_bboxes(civ_d['bbox'], wep_d['bbox']) < WEAPON_PROXIMITY_THRESHOLD:
                                is_armed_in_this_specific_frame = True; break
                        if is_armed_in_this_specific_frame: break
                    if not is_armed_in_this_specific_frame:
                        start_frame_id_for_qwen = frame_meta['frame_id']; break
            if start_frame_id_for_qwen == -1: start_frame_id_for_qwen = 0
    elif yolo_aggregated_evidence.get("soldier_activity_frames", 0) > 0:
        primary_event_type = "Routine Soldier Activity"
        qwen_prompt = "INFO: Predominantly soldier activity observed. Provide a general summary of the activities."
        start_frame_id_for_qwen = 0 
    else:
        primary_event_type = "No Specific Classified Event"
        qwen_prompt = "The video has been processed. No specific pre-defined critical events were detected based on current rules. Perform a general analysis if desired."
        start_frame_id_for_qwen = 0

    if start_frame_id_for_qwen != -1:
        print(f"Primary event '{primary_event_type}' considered to start around frame {start_frame_id_for_qwen}. Collecting frames for Qwen.")
        for frame_meta in all_raw_frames_data:
            if frame_meta['frame_id'] >= start_frame_id_for_qwen:
                if frame_meta['frame_bgr'] is not None:
                    success, encoded_image = cv2.imencode('.jpg', frame_meta['frame_bgr'])
                    if success:
                        frames_for_qwen_output.append({
                            'frame_id': frame_meta['frame_id'],
                            'timestamp': frame_meta['timestamp'],
                            'frame_image_base64': base64.b64encode(encoded_image).decode('utf-8'), 
                            'detections': frame_meta['detections'] 
                        })
                    else:
                        print(f"Warning: Failed to encode frame {frame_meta['frame_id']} for Qwen.")
        
        MAX_QWEN_FRAMES = 50 
        if len(frames_for_qwen_output) > MAX_QWEN_FRAMES:
            print(f"Trimming Qwen frames from {len(frames_for_qwen_output)} to {MAX_QWEN_FRAMES}")
            frames_for_qwen_output = frames_for_qwen_output[:MAX_QWEN_FRAMES]

    if not frames_for_qwen_output and primary_event_type not in ["No Significant Event", "No Specific Classified Event", "Routine Soldier Activity"]:
        print(f"Warning: Event '{primary_event_type}' determined, but no frames were collected for Qwen based on start_frame_id.")
    elif not frames_for_qwen_output and primary_event_type == "Routine Soldier Activity":
        count = 0
        for frame_meta in all_raw_frames_data:
            if frame_meta.get('detections'): 
                success, encoded_image = cv2.imencode('.jpg', frame_meta['frame_bgr'])
                if success:
                    frames_for_qwen_output.append({
                        'frame_id': frame_meta['frame_id'], 'timestamp': frame_meta['timestamp'],
                        'frame_image_base64': base64.b64encode(encoded_image).decode('utf-8'),
                        'detections': frame_meta['detections']
                    })
                    count += 1
                    if count >= 5: break 
        if frames_for_qwen_output:
            print(f"For '{primary_event_type}', sending {len(frames_for_qwen_output)} sample frames to Qwen.")

    return primary_event_type, qwen_prompt, frames_for_qwen_output

# Main analysis function to be called by Flask
# *** MODIFIED FUNCTION SIGNATURE TO ACCEPT yolo_model_path_override ***
def analyze_video_full(video_path_main, yolo_model_path_override=DEFAULT_YOLO_MODEL_PATH): # Added yolo_model_path_override
    global_event_evidence_for_video = {
        "explosion_sound_detected": False, "gunfire_sound_detected": False,
        "fence_tampering_sound_detected": False,
    }

    yamnet_local_evidence, yamnet_overall_results_for_video = process_audio_with_yamnet(video_path_main)
    
    # *** PASS yolo_model_path_override TO process_video_with_yolo ***
    yolo_results_per_frame, yolo_aggregated_counts = process_video_with_yolo(
        video_path_yolo=video_path_main, 
        yolo_model_path_yolo=yolo_model_path_override, # Pass the path here
        yamnet_sound_evidence_for_video=yamnet_local_evidence
    )
    
    all_raw_frames_data_from_yolo = yolo_aggregated_counts.pop("frames_for_qwen_raw", [])

    primary_event, qwen_prompt, qwen_frames_package = determine_primary_event_and_qwen_package(
        yamnet_local_evidence, 
        yolo_aggregated_counts,  
        all_raw_frames_data_from_yolo 
    )

    analysis_summary = {
        "video_path": video_path_main,
        "yamnet_results": yamnet_overall_results_for_video,
        "yolo_detections_summary": yolo_aggregated_counts, 
        "yolo_per_frame_details_count": len(yolo_results_per_frame),
        "primary_event_determined": primary_event,
        "qwen_submission_plan": {
            "prompt": qwen_prompt,
            "frames_count": len(qwen_frames_package),
        }
    }
    return analysis_summary, qwen_prompt, qwen_frames_package

if __name__ == '__main__':
    test_video_path = '/Users/devika/Downloads/main_folder/backend_python/static/videos/video1.mp4' # Example path
    test_yolo_model = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best.pt") # Assuming best.pt is with analyzer

    if os.path.exists(test_video_path) and os.path.exists(test_yolo_model):
        print(f"Testing with video: {test_video_path}")
        print(f"Testing with YOLO model: {test_yolo_model}")
        summary, q_prompt, q_frames = analyze_video_full(test_video_path, yolo_model_path_override=test_yolo_model)
        print("\n--- ANALYSIS SUMMARY (Local Test) ---")
        import json
        print(json.dumps(summary, indent=2))
        print(f"\nQwen Prompt: {q_prompt}")
        print(f"Frames prepared for Qwen: {len(q_frames)}")
        if q_frames:
            print(f"First Qwen frame ID: {q_frames[0]['frame_id']}, Detections: {len(q_frames[0]['detections'])}")
    else:
        if not os.path.exists(test_video_path): print(f"Test video not found: {test_video_path}")
        if not os.path.exists(test_yolo_model): print(f"Test YOLO model not found: {test_yolo_model}")

