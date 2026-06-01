"""Quick pixel analysis of display video"""
import cv2, numpy as np
cap = cv2.VideoCapture("D:/jiemi/Video_20260430154222007.avi")
for idx in range(20):
    ok, f = cap.read()
    if not ok: break
    gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    r200 = float(np.mean(gray > 200)) * 100
    r150 = float(np.mean(gray > 150)) * 100
    r100 = float(np.mean(gray > 100)) * 100
    r050 = float(np.mean(gray > 50)) * 100
    mx = int(gray.max())
    print(f"{idx}: >200={r200:.4f}% >150={r150:.2f}% >100={r100:.1f}% >50={r050:.1f}% max={mx}")
cap.release()
