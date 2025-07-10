import numpy as np
import json
import matplotlib.pyplot as plt
from io import BytesIO
import base64

def generate_histogram(areas, output_path):
    """生成晶粒面积分布直方图"""
    plt.figure(figsize=(10, 6))
    plt.hist(areas, bins=20, alpha=0.7, color='blue')
    plt.title('晶粒面积分布')
    plt.xlabel('面积 (μm²)')
    plt.ylabel('晶粒数量')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(output_path)
    plt.close()
