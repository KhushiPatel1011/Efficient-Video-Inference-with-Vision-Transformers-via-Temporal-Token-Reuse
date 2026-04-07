import cv2
from pathlib import Path


def extract_frames(video_path: Path, output_dir: Path, max_frames=60):
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    frame_count = 0

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        out_path = output_dir / f"frame_{frame_count:04d}.jpg"
        cv2.imwrite(str(out_path), frame)

        frame_count += 1

    cap.release()

    print(f"Extracted {frame_count} frames → {output_dir}")


def main():
    base = Path("data/experiments")

    video_map = [
        ("low_motion", "clip_a"),
        ("low_motion", "clip_b"),
        ("medium_motion", "clip_c"),
        ("medium_motion", "clip_d"),
        ("high_motion", "clip_e"),
        ("high_motion", "clip_f"),
    ]

    for motion, clip in video_map:
        video_path = base / motion / f"{clip}.mp4"
        output_dir = base / motion / clip

        if not video_path.exists():
            print(f"Missing video: {video_path}")
            continue

        print(f"\nProcessing: {video_path}")
        extract_frames(video_path, output_dir)


if __name__ == "__main__":
    main()