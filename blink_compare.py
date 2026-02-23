import cv2
import numpy as np
import mediapipe as mp
import argparse

mp_face_mesh = mp.solutions.face_mesh

# Eye landmark indices for EAR
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]


def euclid(p1, p2):
    return np.linalg.norm(p1 - p2)


def eye_aspect_ratio(pts, idx):
    # EAR = (||p2-p6|| + ||p3-p5||) / (2*||p1-p4||)
    p1, p2, p3, p4, p5, p6 = [pts[i] for i in idx]
    return (euclid(p2, p6) + euclid(p3, p5)) / (2.0 * euclid(p1, p4) + 1e-9)


def blink_rate_from_video(video_path, ear_thresh=0.22, consec_frames=3, show=True, window_name="Blink"):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1 or np.isnan(fps):
        fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = (total_frames / fps) if total_frames > 0 else 0.0

    blinks = 0
    closed_frames = 0

    with mp_face_mesh.FaceMesh(
        static_image_mode=False,
        refine_landmarks=True,
        max_num_faces=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(rgb)

            ear = None
            if res.multi_face_landmarks:
                face = res.multi_face_landmarks[0]
                pts = np.array([(lm.x * w, lm.y * h) for lm in face.landmark], dtype=np.float32)

                left_ear = eye_aspect_ratio(pts, LEFT_EYE)
                right_ear = eye_aspect_ratio(pts, RIGHT_EYE)
                ear = (left_ear + right_ear) / 2.0

                # Blink detection
                if ear < ear_thresh:
                    closed_frames += 1
                else:
                    if closed_frames >= consec_frames:
                        blinks += 1
                    closed_frames = 0

                if show:
                    for idx in LEFT_EYE + RIGHT_EYE:
                        x, y = int(pts[idx][0]), int(pts[idx][1])
                        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            if show:
                cv2.putText(frame, f"{window_name}", (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame, f"Blinks: {blinks}", (20, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                if ear is not None:
                    cv2.putText(frame, f"EAR: {ear:.3f}", (20, 105),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(frame, "Press Q to stop this video", (20, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2)

                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF in (ord('q'), ord('Q')):
                    break

    cap.release()
    cv2.destroyWindow(window_name)

    # If you quit early, duration from CAP_PROP_FRAME_COUNT becomes misleading.
    # For TA demo: let video complete for accurate rate.
    if duration <= 0.0:
        duration = 60.0  # fallback

    rate_per_sec = blinks / max(duration, 1e-6)
    rate_per_min = blinks / max(duration / 60.0, 1e-6)

    return blinks, duration, rate_per_sec, rate_per_min


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--movie", required=True, help="Path to movie video (face visible)")
    parser.add_argument("--reading", required=True, help="Path to reading video (face visible)")
    parser.add_argument("--ear_thresh", type=float, default=0.22, help="EAR threshold")
    parser.add_argument("--consec", type=int, default=3, help="Consecutive frames below threshold to count blink")
    args = parser.parse_args()

    print("\nRunning movie video...")
    mb, md, mps, mpm = blink_rate_from_video(
        args.movie, ear_thresh=args.ear_thresh, consec_frames=args.consec,
        show=True, window_name="Movie Blink Counter"
    )

    print("\nRunning reading video...")
    rb, rd, rps, rpm = blink_rate_from_video(
        args.reading, ear_thresh=args.ear_thresh, consec_frames=args.consec,
        show=True, window_name="Reading Blink Counter"
    )

    print("\n================ FINAL RESULTS ================")
    print(f"Movie   | blinks={mb:3d} | duration={md:6.2f}s | rate={mps:.4f} blinks/sec | {mpm:.2f} blinks/min")
    print(f"Reading | blinks={rb:3d} | duration={rd:6.2f}s | rate={rps:.4f} blinks/sec | {rpm:.2f} blinks/min")

    diff = rps - mps
    if diff > 0:
        print(f"\nObservation: Reading blink rate is HIGHER by {diff:.4f} blinks/sec.")
    else:
        print(f"\nObservation: Reading blink rate is LOWER by {abs(diff):.4f} blinks/sec.")

    print("=============================================\n")

    # Keeps PyCharm console open so you can see results
    input("Press Enter to exit...")


if __name__ == "__main__":
    main()