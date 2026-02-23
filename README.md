# Eye Blink Rate – Computer Vision Prototype

This project is a preliminary computer vision prototype to estimate and compare
eye blink rates while watching a movie versus reading a document using face
landmark detection.

The application uses OpenCV and MediaPipe Face Mesh to detect eye landmarks,
compute Eye Aspect Ratio (EAR), count blinks, and report blink rate (blinks/sec).

## Project Structure

```
Input data/
  blinking_0.mp4
  blinking_1.mp4
  blinking_2.mp4

src/
  blink_compare.py
```

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python src/blink_compare.py --movie "Input data/blinking_0.mp4" --reading "Input data/blinking_1.mp4"
```

Optional parameters:

```bash
python src/blink_compare.py --movie "Input data/blinking_0.mp4" --reading "Input data/blinking_1.mp4" --ear_thresh 0.22 --consec 3
```

## Output

- Live blink counter on video frames  
- Final comparison in the console:
  - Total blinks  
  - Duration (seconds)  
  - Blink rate (blinks/sec and blinks/min)  
  - Observation comparing movie vs reading blink rate  

## Notes

- Ensure your face is clearly visible in the videos.
- Use good lighting to improve detection accuracy.
- Let each video play fully for accurate blink-rate computation.
