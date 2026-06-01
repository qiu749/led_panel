@echo off
cd /d D:\led-visual-inspection
.\venv\Scripts\python.exe -c "import cv2,numpy as np; cap=cv2.VideoCapture('D:/jiemi/Video_20260430154222007.avi'); idx=0
while idx<20:
 ok,f=cap.read()
 if not ok: break
 gray=cv2.cvtColor(f,cv2.COLOR_BGR2GRAY)
 r1=float(np.mean(gray>200));r2=float(np.mean(gray>150));r3=float(np.mean(gray>100))
 r4=float(np.mean(gray>50));mx=int(gray.max())
 print(f'{idx}: >200={r1*100:.4f}% >150={r2*100:.2f}% >100={r3*100:.1f}% >50={r4*100:.1f}% max={mx}')
 idx+=1
cap.release()" > _probe_pixels.txt 2>&1
type _probe_pixels.txt
