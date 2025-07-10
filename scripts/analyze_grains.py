import os
import cv2
import numpy as np
import json
import argparse
from tqdm import tqdm
from skimage import measure, morphology

def main():
    parser = argparse.ArgumentParser(description='烧结 NdFeB 磁体晶粒分析')
    parser.add_argument('--input_dir', default='images', help='输入图像目录')
    parser.add_argument('--output_dir', default='results', help='输出目录')
    parser.add_argument('--pixel_size', type=float, default=0.1, 
                        help='像素尺寸(μm/像素)，默认0.1')
    parser.add_argument('--min_grain_size', type=int, default=50,
                        help='最小晶粒尺寸(像素)，默认50')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    all_results = {}
    
    # 获取所有图像文件
    image_files = [f for f in os.listdir(args.input_dir) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
    
    print(f"开始处理 {len(image_files)} 张SEM图像...")
    
    for img_file in tqdm(image_files):
        img_path = os.path.join(args.input_dir, img_file)
        try:
            # 图像处理流程
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError("无法读取图像")
                
            # 图像预处理
            img = cv2.medianBlur(img, 5)
            img = cv2.equalizeHist(img)
            
            # 阈值分割
            _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 形态学操作
            kernel = np.ones((3,3), np.uint8)
            morph = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
            
            # 连通域分析
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                morph, connectivity=8
            )
            
            # 分析晶粒
            grain_data = []
            for i in range(1, num_labels):  # 跳过背景
                area = stats[i, cv2.CC_STAT_AREA]
                if area < args.min_grain_size:
                    continue
                    
                actual_area = area * (args.pixel_size ** 2)
                grain_data.append({
                    "grain_id": i,
                    "pixel_area": area,
                    "actual_area_um2": actual_area,
                    "centroid_x": centroids[i][0],
                    "centroid_y": centroids[i][1]
                })
            
            # 保存结果
            base_name = os.path.splitext(img_file)[0]
            output_path = os.path.join(args.output_dir, f"{base_name}_results.json")
            with open(output_path, 'w') as f:
                json.dump(grain_data, f, indent=2)
                
            # 可视化结果
            output_img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            for grain in grain_data:
                x, y = int(grain['centroid_x']), int(grain['centroid_y'])
                cv2.circle(output_img, (x, y), 5, (0, 0, 255), -1)
                cv2.putText(output_img, f"{grain['actual_area_um2']:.2f}", (x+10, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            
            cv2.imwrite(os.path.join(args.output_dir, f"{base_name}_visualized.jpg"), output_img)
            
            # 收集统计信息
            areas = [g['actual_area_um2'] for g in grain_data]
            all_results[img_file] = {
                "grain_count": len(areas),
                "mean_area": np.mean(areas) if areas else 0,
                "median_area": np.median(areas) if areas else 0,
                "min_area": np.min(areas) if areas else 0,
                "max_area": np.max(areas) if areas else 0
            }
            
        except Exception as e:
            print(f"处理图像 {img_file} 时出错: {str(e)}")
            all_results[img_file] = {"error": str(e)}
    
    # 保存汇总结果
    with open(os.path.join(args.output_dir, 'analysis_summary.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("分析完成!")

if __name__ == "__main__":
    main()
