"""
Enhanced dataset generator - produces more diverse, distinct training data
Each class gets unique visual features for better discrimination
"""
import os
import json
import random
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter
from typing import List, Dict

BASE_DIR = Path(__file__).parent.parent.parent / "public"
DATASET_DIR = BASE_DIR / "datasets"

# Each class gets UNIQUE shape + color + background combination
CLASSIFICATION_CLASSES = {
    "airplane": {
        "shape": "triangle_up", "color_range": [(0, 80, 200), (50, 150, 255)],  # blues
        "bg_color": (255, 250, 240), "texture": "horizontal_stripes"
    },
    "bird": {
        "shape": "small_circle_body", "color_range": [(200, 50, 50), (255, 100, 100)],  # reds
        "bg_color": (240, 255, 240), "texture": "dots"
    },
    "car": {
        "shape": "rounded_rect", "color_range": [(0, 150, 0), (50, 220, 50)],  # greens
        "bg_color": (245, 245, 255), "texture": "grid"
    },
    "cat": {
        "shape": "circle_ears", "color_range": [(200, 150, 0), (255, 200, 50)],  # orange/gold
        "bg_color": (255, 245, 235), "texture": "vertical_stripes"
    },
    "dog": {
        "shape": "rounded_square", "color_range": [(120, 60, 40), (180, 120, 80)],  # browns
        "bg_color": (235, 245, 255), "texture": "checkerboard"
    },
    "elephant": {
        "shape": "large_oval", "color_range": [(100, 100, 120), (160, 160, 180)],  # grays
        "bg_color": (255, 240, 220), "texture": "crosshatch"
    },
    "flower": {
        "shape": "petals", "color_range": [(200, 50, 150), (255, 100, 200)],  # pink/magenta
        "bg_color": (240, 255, 250), "texture": "concentric"
    },
    "horse": {
        "shape": "tall_rect", "color_range": [(80, 40, 20), (140, 80, 50)],  # dark browns
        "bg_color": (255, 255, 240), "texture": "diagonal_stripes"
    },
    "ship": {
        "shape": "boat_shape", "color_range": [(0, 100, 150), (0, 180, 220)],  # teal/cyan
        "bg_color": (240, 250, 255), "texture": "waves"
    },
    "tree": {
        "shape": "triangle_stack", "color_range": [(0, 100, 0), (30, 180, 30)],  # dark greens
        "bg_color": (255, 255, 230), "texture": "speckled"
    }
}


def random_color(c_range):
    """Generate random color within range"""
    lo, hi = c_range
    return (
        random.randint(lo[0], hi[0]),
        random.randint(lo[1], hi[1]),
        random.randint(lo[2], hi[2])
    )


def draw_background(draw: ImageDraw.ImageDraw, w: int, h: int, texture: str, bg_color, fg_color):
    """Draw distinctive background texture"""
    draw.rectangle([0, 0, w, h], fill=bg_color)
    darker = tuple(max(0, c - 30) for c in fg_color)

    if texture == "horizontal_stripes":
        for y in range(0, h, 8):
            draw.line([(0, y), (w, y)], fill=darker + (30,), width=1)
    elif texture == "vertical_stripes":
        for x in range(0, w, 8):
            draw.line([(x, 0), (x, h)], fill=darker + (30,), width=1)
    elif texture == "diagonal_stripes":
        for i in range(-h, w + h, 10):
            draw.line([(i, 0), (i - h, h)], fill=darker + (25,), width=1)
    elif texture == "grid":
        for y in range(0, h, 16):
            draw.line([(0, y), (w, y)], fill=darker + (20,), width=1)
        for x in range(0, w, 16):
            draw.line([(x, 0), (x, h)], fill=darker + (20,), width=1)
    elif texture == "checkerboard":
        cs = 16
        for y in range(0, h, cs):
            for x in range(0, w, cs):
                if (x // cs + y // cs) % 2 == 0:
                    draw.rectangle([x, y, x + cs, y + cs], fill=darker + (15,))
    elif texture == "dots":
        for _ in range(150):
            rx = random.randint(0, w - 1)
            ry = random.randint(0, h - 1)
            rs = random.randint(1, 3)
            draw.ellipse([rx - rs, ry - rs, rx + rs, ry + rs], fill=darker + (30,))
    elif texture == "waves":
        for y in range(0, h, 12):
            pts = [(x, y + int(3 * np.sin(x / 15))) for x in range(0, w, 4)]
            draw.line(pts, fill=darker + (25,), width=1)
    elif texture == "concentric":
        for r in range(20, max(w, h), 30):
            draw.ellipse([w // 2 - r, h // 2 - r, w // 2 + r, h // 2 + r],
                         outline=darker + (20,), width=1)
    elif texture == "crosshatch":
        for i in range(-h, w + h, 12):
            draw.line([(i, 0), (i - h, h)], fill=darker + (20,), width=1)
            draw.line([(i, 0), (i + h, h)], fill=darker + (20,), width=1)
    elif texture == "speckled":
        for _ in range(300):
            rx = random.randint(0, w - 1)
            ry = random.randint(0, h - 1)
            draw.point((rx, ry), fill=darker)


def draw_shape(draw: ImageDraw.ImageDraw, cx: int, cy: int, size: int, shape_type: str, color, variant: int):
    """Draw distinctive shape per class"""
    if shape_type == "triangle_up":
        # Airplane: upward triangle with flat bottom
        pts = [(cx, cy - size), (cx - size, cy + size // 2), (cx + size, cy + size // 2)]
        draw.polygon(pts, fill=color)
        # Add small wing-like extensions
        if variant % 2 == 0:
            wing_s = size // 3
            draw.polygon([(cx - size - wing_s, cy + size // 4),
                          (cx - size, cy + size // 4),
                          (cx - size, cy - size // 4)], fill=color)
            draw.polygon([(cx + size + wing_s, cy + size // 4),
                          (cx + size, cy + size // 4),
                          (cx + size, cy - size // 4)], fill=color)

    elif shape_type == "small_circle_body":
        # Bird: small circle body + triangle beak
        draw.ellipse([cx - size // 2, cy - size // 2, cx + size // 2, cy + size // 2], fill=color)
        # Beak
        beak = [(cx + size // 2, cy), (cx + size, cy - 5), (cx + size, cy + 5)]
        draw.polygon(beak, fill=color)
        # Eye
        draw.ellipse([cx - 2, cy - size // 3, cx + 3, cy - size // 3 + 5],
                     fill=(0, 0, 0))

    elif shape_type == "rounded_rect":
        # Car: rounded rectangle with windows
        r = size // 4
        draw.rounded_rectangle([cx - size, cy - size // 2, cx + size, cy + size // 2],
                               radius=r, fill=color)
        # Window
        win_c = (min(255, color[0] + 80), min(255, color[1] + 80), min(255, color[2] + 80))
        draw.rounded_rectangle([cx + r // 2, cy - size // 3, cx + size - r, cy],
                               radius=3, fill=win_c)
        # Wheels
        wheel_r = size // 6
        draw.ellipse([cx - size + r, cy + size // 2 - wheel_r,
                      cx - size + r + wheel_r * 2, cy + size // 2 + wheel_r],
                     fill=(40, 40, 40))
        draw.ellipse([cx + size - r - wheel_r * 2, cy + size // 2 - wheel_r,
                      cx + size - r, cy + size // 2 + wheel_r],
                     fill=(40, 40, 40))

    elif shape_type == "circle_ears":
        # Cat: circle + triangular ears
        draw.ellipse([cx - size // 2, cy - size // 2, cx + size // 2, cy + size // 2], fill=color)
        # Ears
        ear_s = size // 3
        draw.polygon([(cx - size // 2, cy - size // 4),
                      (cx - size // 2, cy - size // 2 - ear_s),
                      (cx - size // 4, cy - size // 4)], fill=color)
        draw.polygon([(cx + size // 2, cy - size // 4),
                      (cx + size // 2, cy - size // 2 - ear_s),
                      (cx + size // 4, cy - size // 4)], fill=color)
        # Eyes
        draw.ellipse([cx - size // 5, cy - size // 5, cx - 2, cy - size // 5 + 5],
                     fill=(0, 0, 0))
        draw.ellipse([cx + 2, cy - size // 5, cx + size // 5, cy - size // 5 + 5],
                     fill=(0, 0, 0))

    elif shape_type == "rounded_square":
        # Dog: rounded square with snout
        r = size // 5
        draw.rounded_rectangle([cx - size // 2, cy - size // 2, cx + size // 2, cy + size // 2],
                               radius=r, fill=color)
        # Snout (lighter area at bottom)
        snout_c = (min(255, color[0] + 60), min(255, color[1] + 60), min(255, color[2] + 60))
        draw.ellipse([cx - size // 4, cy, cx + size // 4, cy + size // 3], fill=snout_c)
        # Nose
        draw.ellipse([cx - 3, cy + size // 8, cx + 3, cy + size // 8 + 5], fill=(0, 0, 0))

    elif shape_type == "large_oval":
        # Elephant: large oval with trunk
        draw.ellipse([cx - size, cy - size // 2, cx + size, cy + size // 2], fill=color)
        # Trunk
        draw.line([(cx + size // 2, cy), (cx + size + size // 2, cy + size // 3)],
                  fill=color, width=6)
        # Legs
        for lx in [cx - size // 2, cx - size // 4, cx + size // 4, cx + size // 2]:
            draw.rectangle([lx - size // 8, cy + size // 2, lx + size // 8, cy + size],
                           fill=color)

    elif shape_type == "petals":
        # Flower: center circle + petals
        petal_count = 5 + variant % 3
        petal_r = size // 2
        center_r = size // 5
        for i in range(petal_count):
            angle = (360 / petal_count) * i
            px = cx + int(petal_r * 0.7 * np.cos(np.radians(angle)))
            py = cy + int(petal_r * 0.7 * np.sin(np.radians(angle)))
            draw.ellipse([px - petal_r // 2, py - petal_r // 2,
                          px + petal_r // 2, py + petal_r // 2], fill=color)
        # Center
        draw.ellipse([cx - center_r, cy - center_r, cx + center_r, cy + center_r],
                     fill=(255, 220, 50))

    elif shape_type == "tall_rect":
        # Horse: tall rectangle + head
        draw.rectangle([cx - size // 3, cy - size, cx + size // 3, cy + size], fill=color)
        # Head
        head_s = size // 2
        draw.ellipse([cx + size // 4, cy - size - head_s // 2,
                      cx + size // 4 + head_s, cy - size + head_s // 2], fill=color)
        # Legs
        for lx in [cx - size // 5, cx + size // 5]:
            draw.rectangle([lx - 3, cy + size, lx + 3, cy + size + size // 3], fill=color)

    elif shape_type == "boat_shape":
        # Ship: trapezoid hull + mast
        hull_pts = [(cx - size, cy + size // 3), (cx + size, cy + size // 3),
                    (cx + size * 3 // 4, cy - size // 3), (cx - size * 3 // 4, cy - size // 3)]
        draw.polygon(hull_pts, fill=color)
        # Mast
        draw.line([(cx, cy - size // 3), (cx, cy - size)], fill=color, width=3)
        # Sail
        sail_color = (min(255, color[0] + 50), min(255, color[1] + 50), min(255, color[2] + 50))
        draw.polygon([(cx, cy - size + 5), (cx + size // 2, cy - size // 4), (cx, cy - size // 4)],
                     fill=sail_color)

    elif shape_type == "triangle_stack":
        # Tree: 2-3 stacked triangles + trunk
        layers = 2 + variant % 2
        for i in range(layers):
            layer_size = size - i * size // 3
            offset = -size + i * size // 2
            pts = [(cx, cy + offset - layer_size),
                   (cx - layer_size, cy + offset),
                   (cx + layer_size, cy + offset)]
            draw.polygon(pts, fill=color)
        # Trunk
        draw.rectangle([cx - size // 6, cy + offset, cx + size // 6, cy + offset + size // 3],
                       fill=(101, 67, 33))


def generate_enhanced_image(class_info: dict, variant: int) -> Image.Image:
    """Generate one enhanced synthetic image"""
    w, h = 224, 224
    fg = random_color(class_info["color_range"])

    # Varied size per image
    size = random.randint(50, 90)

    # Slight position variation
    cx = w // 2 + random.randint(-30, 30)
    cy = h // 2 + random.randint(-30, 30)

    # Background with class-specific texture
    img = Image.new("RGB", (w, h))
    draw = ImageDraw.Draw(img, "RGBA")
    draw_background(draw, w, h, class_info["texture"], class_info["bg_color"], fg)

    # Add random noise
    pixels = img.load()
    for _ in range(random.randint(50, 250)):
        x = random.randint(0, w - 1)
        y = random.randint(0, h - 1)
        r, g, b = pixels[x, y]
        noise = random.randint(-20, 20)
        pixels[x, y] = (
            max(0, min(255, r + noise)),
            max(0, min(255, g + noise)),
            max(0, min(255, b + noise))
        )

    # Add secondary decoration (small random shapes for realism)
    draw = ImageDraw.Draw(img, "RGBA")
    deco_color = tuple(max(0, c - 60) for c in class_info["bg_color"])
    for _ in range(random.randint(1, 4)):
        ds = random.randint(5, 15)
        dx = random.randint(ds, w - ds)
        dy = random.randint(ds, h - ds)
        if random.random() < 0.5:
            draw.ellipse([dx - ds, dy - ds, dx + ds, dy + ds], outline=deco_color + (40,), width=1)
        else:
            draw.rectangle([dx - ds, dy - ds, dx + ds, dy + ds], outline=deco_color + (40,), width=1)

    # Draw main shape
    draw = ImageDraw.Draw(img, "RGBA")
    draw_shape(draw, cx, cy, size, class_info["shape"], fg, variant)

    # Subtle blur for realism
    img = img.filter(ImageFilter.GaussianBlur(radius=0.3))

    return img


def generate_enhanced_classification(num_per_class: int = 800):
    """Generate enhanced classification dataset"""
    dataset_path = DATASET_DIR / "classification_enhanced"

    for class_name, class_info in CLASSIFICATION_CLASSES.items():
        class_dir = dataset_path / class_name
        class_dir.mkdir(parents=True, exist_ok=True)

        for i in range(num_per_class):
            img = generate_enhanced_image(class_info, variant=i)
            img.save(class_dir / f"{i:05d}.jpg")

    # Save label map
    label_map = {name: idx for idx, name in enumerate(CLASSIFICATION_CLASSES.keys())}
    with open(dataset_path / "labels.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    # Generate annotations
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

    total = len(annotations)
    print(f"Enhanced classification: {total} images, {len(CLASSIFICATION_CLASSES)} classes "
          f"({num_per_class} per class)")
    return str(dataset_path)


def generate_enhanced_nlp(num_per_sentiment: int = 500):
    """Generate much larger NLP sentiment dataset with diverse patterns"""
    dataset_path = DATASET_DIR / "nlp_enhanced"
    dataset_path.mkdir(parents=True, exist_ok=True)

    # Pattern-based generation for more diversity
    pos_subjects = [
        "产品质量", "包装", "物流速度", "客服态度", "性价比", "设计感",
        "使用体验", "做工", "材质", "功能", "外观", "手感", "音质",
        "画质", "续航", "速度", "稳定性", "安全性", "舒适度", "口感",
        "新鲜度", "售后服务", "配送服务", "购物体验", "品牌信誉"
    ]

    pos_phrases = [
        "非常出色", "超出预期", "令人满意", "值得推荐", "物超所值",
        "一流水平", "无可挑剔", "超级棒", "真心不错", "太赞了",
        "专业水准", "顶级品质", "惊艳到了", "爱不释手", "强烈推荐",
        "业界良心", "精益求精", "满分好评", "再来一份", "回购首选",
        "经典之作", "用心良苦", "细节到位", "品质保证", "诚意满满"
    ]

    neg_subjects = [
        "产品质量", "包装", "物流速度", "客服态度", "性价比", "做工",
        "材质", "功能", "外观", "手感", "音质", "画质", "续航",
        "速度", "稳定性", "安全性", "舒适度", "口感", "新鲜度",
        "售后服务", "配送时效", "商品描述", "购物体验", "退款流程"
    ]

    neg_phrases = [
        "令人失望", "差强人意", "不值这个价", "有明显缺陷", "做工粗糙",
        "完全不值", "骗人的", "太差劲了", "后悔购买了", "一塌糊涂",
        "问题很多", "基本不能用", "严重缩水", "敷衍了事", "投诉到底",
        "瑕疵明显", "根本对不上", "轻飘飘的", "太脆弱了", "次品无疑",
        "粗制滥造", "毫无诚意", "忽悠消费者", "质量堪忧", "态度恶劣"
    ]

    neutral_subjects = [
        "产品", "包装", "物流", "客服", "性价比", "设计", "体验",
        "做工", "材质", "功能", "外观", "手感", "普通产品", "标准款",
        "基础版", "常规配置", "默认设置", "一般水平", "及格线", "入门款"
    ]

    neutral_phrases = [
        "一般般", "还行吧", "不好不坏", "中规中矩", "没什么亮点",
        "能用就行", "基本满意", "说不上好", "及格水平", "凑合用",
        "过得去", "还凑合", "马马虎虎", "差不多", "一般水平",
        "没有惊喜", "基本够用", "谈不上好", "算及格", "不会回购",
        "将就用吧", "还行的", "就那样", "无功无过", "普普通通"
    ]

    sentence_templates = [
        "{subject}真的{phrase}！",
        "这个{subject}，{phrase}。",
        "说实话，{subject}{phrase}。",
        "使用后感觉{subject}{phrase}。",
        "评价一下：{subject}{phrase}。",
        "{subject}方面{phrase}，没办法。",
        "已经用了几天，{subject}{phrase}。",
        "第一次买，{subject}{phrase}。",
        "买给朋友的，反馈{subject}{phrase}。",
        "对比了一下，{subject}{phrase}。",
        "总体来说{subject}{phrase}。",
        "不得不承认{subject}{phrase}。",
        "经历这次，{subject}{phrase}。",
        "之前的期望很高，但{subject}{phrase}。",
        "没想到{subject}{phrase}。",
        "{subject}这一块{phrase}，不必多说。",
        "关于{subject}，我只能说{phrase}。",
        "从{subject}来看，{phrase}。",
        "重点说下{subject}，{phrase}。",
        "老用户了，这次{subject}{phrase}。",
    ]

    sentiment_data = []

    # Generate positive samples
    for i in range(num_per_sentiment):
        subj = random.choice(pos_subjects)
        phrase = random.choice(pos_phrases)
        tmpl = random.choice(sentence_templates)
        text = tmpl.format(subject=subj, phrase=phrase)
        # Occasionally add more context
        if random.random() < 0.3:
            extra = random.choice([
                "五星好评！", "会继续支持！", "下次还来。",
                "已经推荐给朋友了。", "回购无数次了。"
            ])
            text += extra
        sentiment_data.append({"text": text, "label": 1, "sentiment": "positive"})

    # Generate negative samples
    for i in range(num_per_sentiment):
        subj = random.choice(neg_subjects)
        phrase = random.choice(neg_phrases)
        tmpl = random.choice(sentence_templates)
        text = tmpl.format(subject=subj, phrase=phrase)
        if random.random() < 0.3:
            extra = random.choice([
                "不会再买了。", "太坑了！", "差评差评！",
                "谁买谁后悔。", "已经退货了。"
            ])
            text += extra
        sentiment_data.append({"text": text, "label": 0, "sentiment": "negative"})

    # Generate neutral samples
    neutral_texts = []
    for i in range(num_per_sentiment):
        subj = random.choice(neutral_subjects)
        phrase = random.choice(neutral_phrases)
        tmpl = random.choice(sentence_templates)
        text = tmpl.format(subject=subj, phrase=phrase)
        neutral_texts.append({"text": text, "label": 2, "sentiment": "neutral"})

    # Also add opinion review style neutrals
    opinion_templates = [
        "有的地方好有的地方不好，{good_point}不错但{bad_point}差了点。",
        "{good_point}是亮点，但{bad_point}拖后腿了。",
        "整体还行，{good_point}可以，{bad_point}一般。",
        "算不上好也算不上差，{good_point}还不错，就是{bad_point}。",
    ]
    for _ in range(num_per_sentiment // 4):
        text = random.choice(opinion_templates).format(
            good_point=random.choice(pos_subjects),
            bad_point=random.choice(neg_subjects)
        )
        neutral_texts.append({"text": text, "label": 2, "sentiment": "neutral"})

    sentiment_data.extend(neutral_texts)
    random.shuffle(sentiment_data)

    with open(dataset_path / "sentiment.json", "w", encoding="utf-8") as f:
        json.dump(sentiment_data, f, ensure_ascii=False, indent=2)

    pos_count = sum(1 for s in sentiment_data if s["sentiment"] == "positive")
    neg_count = sum(1 for s in sentiment_data if s["sentiment"] == "negative")
    neu_count = sum(1 for s in sentiment_data if s["sentiment"] == "neutral")
    print(f"Enhanced NLP sentiment: {len(sentiment_data)} total "
          f"(pos={pos_count}, neg={neg_count}, neu={neu_count})")
    return str(dataset_path)


if __name__ == "__main__":
    print("Generating enhanced datasets...")
    print("=" * 60)
    generate_enhanced_classification(num_per_class=800)
    generate_enhanced_nlp(num_per_sentiment=500)
    print("Done!")