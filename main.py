#1 Фильтр Гаусса (сглаживание). 1 Автоконтраст

from enum import auto
import cv2
import os
import numpy as np
from datetime import datetime



def gauss(image):
    return cv2.GaussianBlur(image, (5, 5), 0)


def autocontrast(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]
    lo = np.percentile(L, 0.5)
    hi = np.percentile(L, 100 - 0.5)
    L_new = np.clip((L.astype(np.float32) - lo) * 255 / (hi - lo), )
    lab[:, :, 0] = L_new
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def log(filename, status):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    text = f"[{timestamp}] Файл: {filename} | Фильтр: Метод Гаусса + Контраст | Статус: {status} \n"
    with open("Lab2_Media/logs/history.txt", "a", encoding="utf-8") as f:
        f.write(text)


def main():
    filename = input("Какой файл нужно обработать? ")
    path = f"Lab2_Media/data/input/{filename}"
    if not os.path.isfile(path):
        print(f"Ошибка: Файл не найден")
        log(filename, "Ошибка доступа")
        return
    try:
        with open(path, "rb"):
            pass
    except (PermissionError, IOError, OSError):
        print(f"Ошибка: Нет прав на чтение файла {filename}. Обратитесь к администратору")
        log(filename, "Ошибка доступа")
        return
    image = cv2.imread(path)
    task = gauss(autocontrast(image))
    cv2.imwrite("Lab2_Media/data/output/result.jpg", task)
    log(filename, "Успешно")

main()