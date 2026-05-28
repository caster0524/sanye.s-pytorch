"""简化版训练脚本 - 只做分类训练"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from training.train_models import train_classification, train_detection, train_segmentation

if __name__ == "__main__":
    print("开始 ResNet50 分类训练...")
    result = train_classification("resnet50", 10, 5, 4, 0.0005)
    print(f"分类训练完成: {result}")
    
    print("开始 FasterRCNN 检测训练...")
    result = train_detection(epochs=3, batch_size=2)
    print(f"检测训练完成: {result}")
    
    print("开始 DeepLabV3 分割训练...")
    result = train_segmentation(epochs=3, batch_size=2)
    print(f"分割训练完成: {result}")
    
    print("全部训练完成!")