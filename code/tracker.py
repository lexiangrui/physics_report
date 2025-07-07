import cv2
import numpy as np
from ultralytics import YOLO
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable
import logging
from pathlib import Path
import json

# 配置日志
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class TrackerConfig:
    """追踪器配置类"""
    max_size_change: float = 0.3
    size_history_len: int = 5
    confidence_threshold: float = 0.5
    model_path: str = "best.pt"
    
    @classmethod
    def from_json(cls, json_path: str) -> 'TrackerConfig':
        """从JSON文件加载配置"""
        try:
            with open(json_path, 'r') as f:
                config_dict = json.load(f)
            return cls(**config_dict)
        except Exception as e:
            logging.warning(f"加载配置文件失败: {e}，使用默认配置")
            return cls()

class BallTracker:
    def __init__(self, video_path: str, config: Optional[TrackerConfig] = None):
        """
        初始化球体追踪器
        
        Args:
            video_path: 视频文件路径
            config: 追踪器配置，如果为None则使用默认配置
        """
        self.video_path = Path(video_path)
        self.config = config or TrackerConfig()
        
        try:
            self.model = YOLO(self.config.model_path)
        except Exception as e:
            logging.error(f"加载YOLO模型失败: {e}")
            raise
            
        self.trajectory: List[Tuple[float, float]] = []
        self.time_points: List[float] = []
        self.box_sizes: List[List[float]] = []
        self.total_frames: int = 0
        self.fps: float = 0
        self.width: int = 0
        self.height: int = 0
        
        if not self.video_path.exists():
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
            
    def get_median_box_size(self) -> Optional[Tuple[float, float]]:
        """获取历史框的中位数大小"""
        if not self.box_sizes:
            return None
        return tuple(np.median(self.box_sizes, axis=0))
        
    def is_valid_box_size(self, width: float, height: float) -> bool:
        """检查框的大小是否合理"""
        if not self.box_sizes:
            return True
            
        median_size = self.get_median_box_size()
        if not median_size:
            return True
            
        median_width, median_height = median_size
        width_change = abs(width - median_width) / median_width
        height_change = abs(height - median_height) / median_height
        
        return (width_change <= self.config.max_size_change and 
                height_change <= self.config.max_size_change)
        
    def process_frame(self, frame: np.ndarray, current_time: float) -> np.ndarray:
        """处理单帧图像"""
        try:
            results = self.model(frame)[0]
            best_detection = self._get_best_detection(results)
            
            if best_detection:
                x1, y1, x2, y2, score = best_detection
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                width = x2 - x1
                height = y2 - y1
                
                self.trajectory.append((center_x, center_y))
                self.time_points.append(current_time)
                self.box_sizes.append([width, height])
                
                if len(self.box_sizes) > self.config.size_history_len:
                    self.box_sizes.pop(0)
                
                results.boxes.data = results.boxes.data[[
                    i for i, r in enumerate(results.boxes.data.tolist())
                    if r[:4] == [x1, y1, x2, y2]
                ]]
                
                annotated_frame = results.plot()
                return annotated_frame
            
            return frame
            
        except Exception as e:
            logging.error(f"处理帧时发生错误: {e}")
            return frame
            
    def _get_best_detection(self, results) -> Optional[Tuple]:
        """获取最佳检测结果"""
        best_detection = None
        best_score = 0
        
        for r in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = r
            if (score > self.config.confidence_threshold and 
                int(class_id) == 0):
                width = x2 - x1
                height = y2 - y1
                
                if self.is_valid_box_size(width, height) and score > best_score:
                    best_score = score
                    best_detection = (x1, y1, x2, y2, score)
                    
        return best_detection
        
    def run(self):
        """运行视频处理 - 初始化并调用处理循环"""
        try:
            cap = cv2.VideoCapture(str(self.video_path))
            if not cap.isOpened():
                raise RuntimeError("无法打开视频文件")

        
            self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = cap.get(cv2.CAP_PROP_FPS)
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            logging.info(f"视频信息: {self.width}x{self.height} @ {self.fps:.2f} FPS, 共 {self.total_frames} 帧")

        except Exception as e:
            logging.error(f"初始化视频处理时发生错误: {e}")
            raise
        finally:
            if cap and cap.isOpened():
                cap.release()

    def process_video_stream(self, progress_callback: Optional[Callable[[int, int], None]] = None):
        """处理视频流，保存带追踪框的视频，并收集数据点。

        Args:
            progress_callback: 一个可选的回调函数，接收 (当前帧号, 总帧数)。
        """
        cap = None
        out = None
        try:
            cap = cv2.VideoCapture(str(self.video_path))
            if not cap.isOpened():
                raise RuntimeError("无法重新打开视频文件进行处理")

            if not self.width or not self.height or not self.fps or not self.total_frames:
                self.run() 
                if not self.width: 
                     raise RuntimeError("视频信息未能成功加载")

            output_path = self.video_path.with_name(
                f"{self.video_path.stem}_tracked{self.video_path.suffix}")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, self.fps, (self.width, self.height))
            if not out.isOpened():
                raise RuntimeError(f"无法创建输出视频文件: {output_path}")

            frame_count = 0
            self.trajectory = [] 
            self.time_points = []
            self.box_sizes = []

            logging.info("开始逐帧处理视频...")
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    logging.info("视频帧读取完毕或发生错误")
                    break

                current_time = frame_count / self.fps
                processed_frame = self.process_frame(frame, current_time) 
                out.write(processed_frame)

                frame_count += 1

                if progress_callback:
                    try:
                        progress_callback(frame_count, self.total_frames)
                    except Exception as cb_e:
                        logging.warning(f"进度回调函数出错: {cb_e}")

            logging.info(f"视频处理完成，共处理 {frame_count} 帧。")

        except Exception as e:
            logging.error(f"处理视频流时发生错误: {e}")
            import traceback
            traceback.print_exc()
            raise 
        finally:
            if cap and cap.isOpened():
                cap.release()
            if out and out.isOpened():
                out.release()
                logging.info(f"处理后的视频已保存至: {output_path}")
            else:
                 logging.warning(f"输出视频未能成功保存: {output_path}")

    def _save_trajectory_data(self):
        """保存轨迹数据到CSV，不创建matplotlib图形窗口"""
        csv_path = self.video_path.with_name(f"{self.video_path.stem}_trajectory_data.csv")
        
        try:
            with open(csv_path, 'w') as f:
                f.write("time,x_position\n")
                
                for t, (x, _) in zip(self.time_points, self.trajectory):
                    f.write(f"{t:.4f},{x:.2f}\n")
                           
            logging.info(f"轨迹数据已保存至: {csv_path}")
            
        except Exception as e:
            logging.error(f"保存轨迹数据时发生错误: {e}")
            
    def _save_statistics(self):
        """保存统计信息"""
        stats_path = self.video_path.with_name(f"{self.video_path.stem}_stats.txt")
        try:
            with open(stats_path, 'w') as f:
                f.write("运动统计信息:\n")
                f.write(f"总记录时间: {self.time_points[-1]:.2f} 秒\n")
                f.write(f"记录的数据点数量: {len(self.trajectory)}\n")
                f.write(f"平均采样率: {len(self.trajectory)/self.time_points[-1]:.2f} Hz\n")
                
                x_coords = [p[0] for p in self.trajectory]
                y_coords = [p[1] for p in self.trajectory]
                f.write(f"X方向运动范围: {min(x_coords):.2f} - {max(x_coords):.2f} 像素\n")
                f.write(f"Y方向运动范围: {min(y_coords):.2f} - {max(y_coords):.2f} 像素\n")
                
                if self.box_sizes:
                    diameters = [(w + h) / 2 for w, h in self.box_sizes]
                    avg_diameter = sum(diameters) / len(diameters)
                    max_diameter = max(diameters)
                    min_diameter = min(diameters)
                    f.write(f"球体像素直径: 平均 {avg_diameter:.2f} 像素 (范围: {min_diameter:.2f} - {max_diameter:.2f} 像素)\n")
                
            logging.info(f"统计信息已保存至: {stats_path}")
            
        except Exception as e:
            logging.error(f"保存统计信息时发生错误: {e}")
    
    def plot_trajectory(self):
        """保存轨迹数据、统计信息"""
        if not self.trajectory:
            logging.warning("没有检测到球的轨迹，无法保存数据或绘图")
            return

        try:
            self._save_trajectory_data()
            self._save_statistics()

        except Exception as e:
            logging.error(f"保存轨迹数据时出错: {e}")
            import traceback
            traceback.print_exc()