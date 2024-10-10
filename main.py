import os
import cv2
import numpy as np
import easyocr
import time
import subprocess

# 初始化 easyocr 识别器
reader = easyocr.Reader(['en'])

def take_screenshot():
    # 使用 ADB 截图并直接转换为内存中的图像对象
    result = subprocess.run(["adb", "exec-out", "screencap", "-p"], stdout=subprocess.PIPE)
    image = np.frombuffer(result.stdout, np.uint8)
    screenshot = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return screenshot

def preprocess_image(image):
    # 确保读取到的图像不是 None
    if image is None:
        raise FileNotFoundError("无法读取图像文件")

    # 转换为灰度图像（确保图像是单通道的）
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 去噪声 - 高斯模糊
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # 二值化处理
    _, thresholded = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 形态学处理 - 膨胀和腐蚀（让数字的边缘更加清晰）
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    processed_image = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)

    return processed_image

def crop_numbers(image):
    height, width, _ = image.shape

    # 裁剪左边数字区域
    left_number_area = image[int(height / 4):int(height / 3), int(width * 0.1):int(width * 0.4)]

    # 裁剪右边数字区域
    right_number_area = image[int(height / 4):int(height / 3), int(width * 0.6):int(width * 0.9)]

    return left_number_area, right_number_area

def recognize_numbers_with_easyocr(image):
    # 使用 easyocr 识别
    _, image_buffer = cv2.imencode('.png', image)
    image_bytes = image_buffer.tobytes()
    result = reader.readtext(image_bytes, detail=0)
    return result[0] if result else ""  # 返回识别结果，若无结果则返回空字符串

def draw_symbol_on_screen(symbol, screen_width=1080, screen_height=1920):
    # 计算屏幕中下部位置的起始坐标
    start_x = 2 * screen_width // 3
    start_y = int(screen_height * 0.75)  # 纵坐标在屏幕高度的 75% 处，位于中下部

    # 设置符号的长度为约 20 像素
    offset = 30

    # 持续时间
    duration = 10

    if symbol == '>':
        # 绘制大于号的折线，分为两段（单次滑动中实现两个线段）
        os.system(
            f"adb shell input swipe {start_x - offset} {start_y - offset} {start_x + offset} {start_y + offset} {duration} && "
            f"adb shell input swipe {start_x + offset} {start_y + offset} {start_x - offset} {start_y + 2 * offset} {duration}"
        )
    elif symbol == '<':
        # 绘制小于号的折线，分为两段
        os.system(
            f"adb shell input swipe {start_x + offset} {start_y - offset} {start_x - offset} {start_y + offset} {duration} && "
            f"adb shell input swipe {start_x - offset} {start_y + offset} {start_x + offset} {start_y + 2 * offset} {duration}"
        )

def images_are_different(img1, img2, threshold=100):
    # 计算两张图片的差异
    diff = cv2.absdiff(img1, img2)
    diff_sum = diff.sum()

    # 如果差异超过阈值，则认为两张图片不同
    return diff_sum > threshold

if __name__ == "__main__":
    previous_left = None
    previous_right = None

    while True:
        # 获取当前截图
        screenshot = take_screenshot()

        # 裁剪左右数字区域
        left_number_area, right_number_area = crop_numbers(screenshot)

        # 预处理图像
        left_processed = preprocess_image(left_number_area)
        right_processed = preprocess_image(right_number_area)

        # 检查左右数字区域是否发生变化
        if previous_left is None or previous_right is None or \
                images_are_different(previous_left, left_processed) or images_are_different(previous_right, right_processed):
            # 更新 previous_left 和 previous_right
            previous_left = left_processed
            previous_right = right_processed

            # 使用 easyocr 进行识别
            left_number = recognize_numbers_with_easyocr(left_processed)
            right_number = recognize_numbers_with_easyocr(right_processed)

            print("左边数字:", left_number)
            print("右边数字:", right_number)

            # 比较两个数字并绘制符号
            try:
                if int(left_number) > int(right_number):
                    print("大")
                    draw_symbol_on_screen('>')
                elif int(left_number) < int(right_number):
                    print("小")
                    draw_symbol_on_screen('<')
                else:
                    print("相等")
            except ValueError:
                print("识别到无效数字，跳过本次操作。")

        # 延迟一段时间，避免频繁截图
        time.sleep(0.05)