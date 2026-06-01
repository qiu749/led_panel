@echo off
cd /d D:\led-visual-inspection
.\venv\Scripts\python.exe -c "import cv2; cap=cv2.VideoCapture('D:/jiemi/Video_20260430154222007.avi'); n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps=cap.get(cv2.CAP_PROP_FPS); w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); cap.release(); print(f'{n},{fps:.2f},{w}x{h}')" > _vinfo.txt 2>&1
type _vinfo.txt
