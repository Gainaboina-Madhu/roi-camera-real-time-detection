import cv2
import torch
import sys
from models.experimental import attempt_load
from utils.general import non_max_suppression

class YoloV7:

    def __init__(self, weights):
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = attempt_load(weights, map_location=self.device)
            self.model.eval()
            print(f"Model loaded on {self.device}")
            print(f"Classes: {self.model.names}")  # shows your 7 classes

        except Exception as e:
            _, msg, tb = sys.exc_info()
            print(f"Error line {tb.tb_lineno}: {msg}")

    def inference(self, frame):
        try:
            # Prepare frame
            img = cv2.cvtColor(cv2.resize(frame, (640, 640)), cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            tensor = tensor.to(self.device)

            # Run model
            with torch.no_grad():
                pred = non_max_suppression(self.model(tensor)[0], 0.25, 0.45)[0]

            # Draw boxes on frame
            result = frame.copy()
            if pred is not None:
                for *xyxy, conf, cls in pred:
                    x1, y1, x2, y2 = map(int, xyxy)
                    label = self.model.names[int(cls)]
                    print(f"Detected: {label} | Confidence: {conf:.2f}")
                    cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(result, f"{label} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            return result

        except Exception as e:
            _, msg, tb = sys.exc_info()
            print(f"Error line {tb.tb_lineno}: {msg}")
            return frame


if __name__ == "__main__":
    try:
        #  Change video path to your video
        obj = YoloV7("best.pt")
        cap = cv2.VideoCapture("C:\\Users\\sunil\\Downloads\\ROI_EVENT\\videos\\car_bike.mp4")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            result = obj.inference(frame)

            cv2.imshow("YOLOv7 Test", result)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

    except Exception as e:
        _, msg, tb = sys.exc_info()
        print(f"Error line {tb.tb_lineno}: {msg}")