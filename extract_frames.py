import cv2
import os
import numpy as np

def extract_frames(video_path, output_dir, num_frames=1000):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"视频总帧数: {total_frames}")
    
    if total_frames == 0:
        print("视频为空或读取失败")
        return
        
    # 计算需要抽取的帧的索引（均匀分布）
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    count = 0
    saved = 0
    
    # 转换为集合以提高查找速度
    target_indices = set(indices)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if count in target_indices:
            # 获取文件名格式：video名称_frame_索引.jpg
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            out_path = os.path.join(output_dir, f"{base_name}_frame_{count:05d}.jpg")
            cv2.imencode('.jpg', frame)[1].tofile(out_path) # 支持中文路径
            saved += 1
            if saved % 100 == 0:
                print(f"已提取 {saved}/{num_frames} 张图片")
                
        count += 1
        
        if saved >= num_frames:
            break
            
    cap.release()
    print(f"完成！共提取 {saved} 张图片到目录: {output_dir}")

if __name__ == "__main__":
    video_file = r"D:\jiemi\Video_20260421163145108.avi"
    output_directory = r"D:\jiemi\extracted_frames"
    extract_frames(video_file, output_directory, 1000)
