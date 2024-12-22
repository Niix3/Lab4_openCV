import cv2
import numpy as np


def read_image(path):
    image = cv2.imread(path)
    if image is None:
        print("Файл не найден")
    return image


def convert_to_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def threshold_image(image, value):
    """
    Зануляем (стали белыми) все точки с интенсивностью меньше value
    :param image: черно-белое изображение
    :param value: порог
    :return: новое изображение после порогового преобразования
    """
    ret, new_image = cv2.threshold(image, value, 255, cv2.THRESH_BINARY)
    print(ret)
    return new_image


def main():
    image = read_image("123.jpg")
    gray_image = convert_to_gray(image)
    thresh = threshold_image(gray_image, 150)

    # cv2.imshow("Original", image)
    # cv2.waitKey(0)
    cv2.imshow("Gray", gray_image)
    cv2.waitKey(0)
    cv2.imshow("Thresh", thresh)
    cv2.waitKey(0)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
