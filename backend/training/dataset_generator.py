"""
训练数据集生成器
为项目的5个任务（分类/检测/分割/NLP/生成）生成训练数据集
"""
import os
import json
import random
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from typing import List, Dict, Tuple, Optional

BASE_DIR = Path(__file__).parent.parent.parent.parent / "public"
DATASET_DIR = BASE_DIR / "datasets"

# 分类类别 (精选10个常见类别用于微调)
CLASSIFICATION_CLASSES = [
    "airplane", "bird", "car", "cat", "dog",
    "elephant", "flower", "horse", "ship", "tree"
]

# COCO 检测类别 (精选10个)
DETECTION_CLASSES = [
    "person", "car", "dog", "cat", "bicycle",
    "chair", "bottle", "laptop", "cell phone", "book"
]

# Pascal VOC 分割类别
SEGMENTATION_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]


def generate_synthetic_image(width: int = 224, height: int = 224, shape: str = "circle") -> Image.Image:
    """生成简单合成图像用于训练"""
    img = Image.new("RGB", (width, height), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    
    color = (
        random.randint(30, 220),
        random.randint(30, 220),
        random.randint(30, 220)
    )
    
    cx, cy = width // 2, height // 2
    size = random.randint(30, min(width, height) // 3)
    
    if shape == "circle":
        draw.ellipse([cx - size, cy - size, cx + size, cy + size], fill=color)
    elif shape == "rectangle":
        draw.rectangle([cx - size, cy - size, cx + size, cy + size], fill=color)
    elif shape == "triangle":
        draw.polygon([(cx, cy - size), (cx - size, cy + size), (cx + size, cy + size)], fill=color)
    elif shape == "cross":
        draw.line([(cx - size, cy), (cx + size, cy)], fill=color, width=8)
        draw.line([(cx, cy - size), (cx, cy + size)], fill=color, width=8)
    elif shape == "star":
        # 简单五角星
        pts = []
        for i in range(5):
            angle = -90 + i * 72
            outer_x = cx + int(size * 0.95 * np.cos(np.radians(angle)))
            outer_y = cy + int(size * 0.95 * np.sin(np.radians(angle)))
            pts.append((outer_x, outer_y))
            inner_angle = angle + 36
            inner_x = cx + int(size * 0.4 * np.cos(np.radians(inner_angle)))
            inner_y = cy + int(size * 0.4 * np.sin(np.radians(inner_angle)))
            pts.append((inner_x, inner_y))
        draw.polygon(pts, fill=color)
    elif shape == "donut":
        draw.ellipse([cx - size, cy - size, cx + size, cy + size], fill=color)
        inner = size * 2 // 3
        draw.ellipse([cx - inner, cy - inner, cx + inner, cy + inner], fill=(240, 240, 240))
    
    # 添加随机噪声
    pixels = img.load()
    for _ in range(200):
        x = random.randint(0, width - 1)
        y = random.randint(0, height - 1)
        noise = random.randint(-15, 15)
        r, g, b = pixels[x, y]
        pixels[x, y] = (
            max(0, min(255, r + noise)),
            max(0, min(255, g + noise)),
            max(0, min(255, b + noise))
        )
    
    return img


def generate_classification_dataset(num_per_class: int = 100) -> str:
    """生成图像分类数据集"""
    dataset_path = DATASET_DIR / "classification"
    
    # 每个类别对应不同图形
    class_shapes = {
        "airplane": "triangle", "bird": "cross", "car": "rectangle",
        "cat": "circle", "dog": "star", "elephant": "circle",
        "flower": "star", "horse": "triangle", "ship": "rectangle", "tree": "triangle"
    }
    
    for class_idx, class_name in enumerate(CLASSIFICATION_CLASSES):
        class_dir = dataset_path / class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        
        shape = class_shapes.get(class_name, "circle")
        for i in range(num_per_class):
            img = generate_synthetic_image(224, 224, shape)
            img.save(class_dir / f"{i:05d}.jpg")
    
    # 保存类标签映射
    label_map = {name: idx for idx, name in enumerate(CLASSIFICATION_CLASSES)}
    with open(dataset_path / "labels.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    
    # 生成标注文件
    annotations = []
    for class_name in CLASSIFICATION_CLASSES:
        class_dir = dataset_path / class_name
        for img_file in sorted(class_dir.glob("*.jpg")):
            annotations.append({
                "image": str(img_file.relative_to(dataset_path)),
                "label": label_map[class_name],
                "class_name": class_name
            })
    
    with open(dataset_path / "annotations.json", "w", encoding="utf-8") as f:
        json.dump(annotations, f, ensure_ascii=False, indent=2)
    
    print(f"  分类数据集: {len(annotations)} 样本, {len(CLASSIFICATION_CLASSES)} 类别")
    return str(dataset_path)


def generate_detection_dataset(num_images: int = 200) -> str:
    """生成目标检测数据集"""
    dataset_path = DATASET_DIR / "detection"
    images_dir = dataset_path / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    annotations_json = {"images": [], "annotations": [], "categories": []}
    
    for cat_id, class_name in enumerate(DETECTION_CLASSES, 1):
        annotations_json["categories"].append({
            "id": cat_id, "name": class_name
        })
    
    shapes = ["circle", "rectangle", "triangle", "cross", "star"]
    
    for img_id in range(num_images):
        img = Image.new("RGB", (512, 512), color=(240, 240, 240))
        draw = ImageDraw.Draw(img)
        
        img_file = f"{img_id:05d}.jpg"
        img.save(images_dir / img_file)
        
        annotations_json["images"].append({
            "id": img_id + 1,
            "file_name": f"images/{img_file}",
            "width": 512,
            "height": 512
        })
        
        # 每个图像 2-4 个物体
        num_objects = random.randint(2, 4)
        for _ in range(num_objects):
            cat_id = random.randint(1, len(DETECTION_CLASSES))
            size = random.randint(30, 120)
            x = random.randint(10, 490 - size)
            y = random.randint(10, 490 - size)
            
            color = (
                random.randint(50, 200),
                random.randint(50, 200),
                random.randint(50, 200)
            )
            shape = random.choice(shapes)
            
            if shape == "circle":
                draw.ellipse([x, y, x + size, y + size], fill=color)
            elif shape == "rectangle":
                draw.rectangle([x, y, x + size, y + size], fill=color)
            else:
                draw.ellipse([x, y, x + size, y + size], fill=color)
            
            annotations_json["annotations"].append({
                "id": len(annotations_json["annotations"]) + 1,
                "image_id": img_id + 1,
                "category_id": cat_id,
                "bbox": [x, y, size, size],
                "area": size * size
            })
        
        img.save(images_dir / img_file)
    
    with open(dataset_path / "annotations.json", "w", encoding="utf-8") as f:
        json.dump(annotations_json, f, ensure_ascii=False, indent=2)
    
    print(f"  检测数据集: {num_images} 图像, {len(annotations_json['annotations'])} 标注")
    return str(dataset_path)


def generate_segmentation_dataset(num_images: int = 100) -> str:
    """生成图像分割数据集"""
    dataset_path = DATASET_DIR / "segmentation"
    images_dir = dataset_path / "images"
    masks_dir = dataset_path / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    for img_id in range(num_images):
        img = Image.new("RGB", (256, 256), color=(240, 240, 240))
        mask = Image.new("L", (256, 256), color=0)  # 0 = background
        draw = ImageDraw.Draw(img)
        mask_draw = ImageDraw.Draw(mask)
        
        # 1-3 个分割对象
        num_objects = random.randint(1, 3)
        for _ in range(num_objects):
            class_id = random.randint(1, len(SEGMENTATION_CLASSES) - 1)
            size = random.randint(40, 100)
            cx = random.randint(size, 256 - size)
            cy = random.randint(size, 256 - size)
            
            color = (
                random.randint(50, 200),
                random.randint(50, 200),
                random.randint(50, 200)
            )
            
            # 绘制形状
            draw.ellipse([cx - size, cy - size, cx + size, cy + size], fill=color)
            mask_draw.ellipse([cx - size, cy - size, cx + size, cy + size], fill=class_id)
        
        img.save(images_dir / f"{img_id:05d}.jpg")
        mask.save(masks_dir / f"{img_id:05d}.png")
    
    # 保存类别信息
    with open(dataset_path / "classes.json", "w", encoding="utf-8") as f:
        json.dump(SEGMENTATION_CLASSES, f, ensure_ascii=False, indent=2)
    
    print(f"  分割数据集: {num_images} 图像+掩码, {len(SEGMENTATION_CLASSES)} 类别")
    return str(dataset_path)


def generate_nlp_dataset(num_samples: int = 500) -> str:
    """生成NLP训练数据集（情感分析和文本生成）"""
    dataset_path = DATASET_DIR / "nlp"
    dataset_path.mkdir(parents=True, exist_ok=True)
    
    # 中文情感分析数据集
    positive_texts = [
        "这个产品非常好用，强烈推荐给大家！",
        "质量很棒，性价比很高，下次还会来买。",
        "服务态度很好，物流也快，非常满意。",
        "太喜欢了，超出预期，给五星好评！",
        "设计精美，功能强大，使用体验极佳。",
        "已经推荐给朋友了，大家都说很不错。",
        "第二次购买了，品质一如既往的好。",
        "包装很用心，产品完美无损，赞！",
        "用了几天了，效果显著，很满意。",
        "客服很耐心，解答很详细，体验很好。",
        "这个价格能买到这么好的东西，太值了。",
        "发货速度超级快，第二天就到了。",
        "做工精细，细节处理得很好。",
        "外观漂亮，手感舒适，爱不释手。",
        "真的是良心产品，用心做的。",
        "买给父母的，他们非常喜欢。",
        "比实体店便宜好多，质量一样好。",
        "物流很快，包装完好，很满意的一次购物。",
        "性能稳定，运行流畅，大品牌值得信赖。",
        "零食很好吃，口感酥脆，会回购的。",
        "衣服面料舒服，尺码标准，很合身。",
        "这本书内容很丰富，学到了不少知识。",
        "耳机音质很好，降噪效果棒。",
        "手机拍照效果惊艳，电池续航也不错。",
        "鞋子穿起来很舒服，走路不累。",
    ]
    
    negative_texts = [
        "产品质量太差了，用了两天就坏了。",
        "非常失望，和描述完全不符，欺骗消费者。",
        "物流太慢了，等了半个月才到。",
        "售后服务态度恶劣，完全不解决问题。",
        "性价比低，不值得购买，后悔了。",
        "包装简陋，收到时已经破损了。",
        "味道很奇怪，不敢吃了，直接扔了。",
        "尺寸不对，穿了不合适还不能退。",
        "做工粗糙，细节很不到位。",
        "客服一直踢皮球，问题得不到解决。",
        "功能很多都有问题，像半成品。",
        "比图片差远了，照骗太过分了。",
        "噪音很大，影响休息，打算退货。",
        "电池不耐用，充满电半天就没了。",
        "质量差还贵，完全不如同价位的其他品牌。",
        "有异味，散了半个月都没去掉。",
        "商家发错了货，还不承认错误。",
        "软件bug很多，经常闪退崩溃。",
        "说是正品但到手一眼假，上当了。",
        "拉链没拉几次就坏了，质量堪忧。",
        "说明书全是英文的，根本看不懂。",
        "掉色严重，洗了一次就没法穿了。",
        "操作太复杂了，老人完全不会用。",
        "送了赠品但是坏的，还不如不送。",
        "下完单才告知没货，浪费感情。",
    ]
    
    neutral_texts = [
        "产品还行吧，不好不坏，能用。",
        "一般般，没有想象中的好，也没有很差。",
        "正常收到了，没什么特别的感觉。",
        "普普通通，对得起这个价格。",
        "还可以，基本功能都正常。",
        "没有什么惊喜，跟普通的一样。",
        "用着还行，没发现什么问题。",
        "东西还行，但包装有点简单。",
        "收到货了，看起来还行吧。",
        "就这样吧，懒得退换了。",
        "标准产品，没有什么特别的。",
        "能用就行，不期望太多。",
        "还行，但不会再买第二个了。",
        "还不错，给个中评。",
    ]
    
    sentiment_data = []
    for text in positive_texts[:num_samples // 3]:
        sentiment_data.append({"text": text, "label": 1, "sentiment": "positive"})
    for text in negative_texts[:num_samples // 3]:
        sentiment_data.append({"text": text, "label": 0, "sentiment": "negative"})
    for text in neutral_texts[:num_samples // 3]:
        sentiment_data.append({"text": text, "label": 2, "sentiment": "neutral"})
    
    random.shuffle(sentiment_data)
    
    with open(dataset_path / "sentiment.json", "w", encoding="utf-8") as f:
        json.dump(sentiment_data, f, ensure_ascii=False, indent=2)
    
    # 文本生成训练数据（提示词-回复对）
    text_gen_prompts = [
        {"prompt": "人工智能是", "completion": "人工智能是计算机科学的一个分支，旨在创建能够执行通常需要人类智能的任务的系统。这些任务包括学习、推理、问题解决、感知和语言理解等。"},
        {"prompt": "深度学习的核心思想是", "completion": "深度学习的核心思想是通过多层神经网络自动学习数据的层次化特征表示，从原始输入逐步提取越来越抽象的概念，最终完成复杂的模式识别和决策任务。"},
        {"prompt": "PyTorch是一个", "completion": "PyTorch是一个开源的深度学习框架，由Facebook的AI研究团队开发。它以动态计算图为核心特性，提供了灵活的张量计算和自动微分机制，是目前最受欢迎的深度学习框架之一。"},
        {"prompt": "卷积神经网络（CNN）主要用于", "completion": "卷积神经网络（CNN）主要用于图像识别和计算机视觉任务。它通过卷积层自动提取图像的局部特征，然后通过池化层降低维度，最终通过全连接层完成分类或回归任务。"},
        {"prompt": "自然语言处理（NLP）的目标是", "completion": "自然语言处理（NLP）的目标是让计算机能够理解、解释和生成人类语言。它涵盖了文本分类、情感分析、机器翻译、问答系统、文本摘要等多个应用领域。"},
        {"prompt": "迁移学习的优势在于", "completion": "迁移学习的优势在于可以将在大型数据集上预训练的模型知识迁移到新的任务上，大大减少了训练所需的数据量和计算资源，同时通常能够获得更好的性能。"},
        {"prompt": "目标检测的任务是", "completion": "目标检测的任务是在图像中同时定位和识别多个物体，输出每个物体的边界框位置和类别标签。常用的方法包括Faster R-CNN、YOLO和SSD等。"},
        {"prompt": "图像分割与目标检测的区别是", "completion": "图像分割与目标检测的区别在于：目标检测输出边界框，而图像分割为每个像素分配一个类别标签，提供更精细的像素级理解。语义分割区分不同类别，实例分割还区分同一类别的不同个体。"},
        {"prompt": "GPU在深度学习中的作用是", "completion": "GPU在深度学习中的作用是加速大规模矩阵运算。由于神经网络训练涉及大量并行的矩阵乘法操作，GPU的数千个核心可以同时处理这些运算，通常能将训练速度提升10-100倍。"},
        {"prompt": "数据增强是", "completion": "数据增强是一种通过对训练数据进行随机变换（如旋转、翻转、缩放、颜色抖动等）来增加数据多样性的技术。它能有效防止过拟合，提高模型的泛化能力，尤其在数据量有限的情况下非常有用。"},
        {"prompt": "损失函数的作用是", "completion": "损失函数的作用是衡量模型预测结果与真实标签之间的差异程度。它提供了优化目标，训练过程就是通过反向传播不断调整模型参数以最小化损失函数值的过程。"},
        {"prompt": "过拟合是指", "completion": "过拟合是指模型在训练数据上表现很好，但在未见过的测试数据上表现很差的现象。这通常是因为模型过于复杂，学习到了训练数据中的噪声而非真正的模式。解决方法包括正则化、Dropout、早停和数据增强等。"},
    ]
    
    with open(dataset_path / "text_generation.json", "w", encoding="utf-8") as f:
        json.dump(text_gen_prompts, f, ensure_ascii=False, indent=2)
    
    print(f"  情感分析数据: {len(sentiment_data)} 样本")
    print(f"  文本生成数据: {len(text_gen_prompts)} 提示词-回复对")
    return str(dataset_path)


def generate_all_datasets():
    """生成所有训练数据集"""
    print("正在生成训练数据集...")
    print("=" * 50)
    
    classification_path = generate_classification_dataset(num_per_class=100)
    detection_path = generate_detection_dataset(num_images=200)
    segmentation_path = generate_segmentation_dataset(num_images=100)
    nlp_path = generate_nlp_dataset(num_samples=200)
    
    print("=" * 50)
    print("所有数据集生成完成！")
    print(f"数据集根目录: {DATASET_DIR}")
    
    return {
        "classification": classification_path,
        "detection": detection_path,
        "segmentation": segmentation_path,
        "nlp": nlp_path
    }


if __name__ == "__main__":
    generate_all_datasets()