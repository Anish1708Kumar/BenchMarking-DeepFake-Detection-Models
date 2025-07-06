import cv2
import os

def extract_frames(video_path, output_dir, every_n_frames=10):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    vidcap = cv2.VideoCapture(video_path)
    count = 0
    frame_idx = 0
    success, image = vidcap.read()
    while success:
        if count % every_n_frames == 0:
            frame_name = f"{video_name}_frame{frame_idx:04d}.jpg"
            cv2.imwrite(os.path.join(output_dir, frame_name), image)
            frame_idx += 1
        success, image = vidcap.read()
        count += 1
    vidcap.release()

# Example usage:
# For all original (real) videos:
for fname in os.listdir('.'):
    if fname.endswith('-original.mov'):
        extract_frames(fname, 'real_frames', every_n_frames=10)
# For all deepfake (fake) videos in higher_quality/lower_quality:
for subdir in ['higher_quality', 'lower_quality']:
    for fname in os.listdir(subdir):
        if fname.endswith('.mov'):
            extract_frames(os.path.join(subdir, fname), 'fake_frames', every_n_frames=10)
