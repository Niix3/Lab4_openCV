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
    ret, new_image = cv2.threshold(image, value, 255, cv2.THRESH_BINARY)
    return new_image


def my_filter(image):
    ker = 1 / 25 * np.ones((5, 5))
    filtered = cv2.filter2D(image, ddepth=-1, kernel=ker)
    return filtered


def show(name, image):
    cv2.imshow(name, image)
    cv2.waitKey(0)


def find_edges(image, method="Canny"):
    if method == "Canny":
        return cv2.Canny(image, 100, 200)
    elif method == "Laplas":
        return cv2.Laplacian(image, cv2.CV_64F)
    elif method == "Sobel":
        return cv2.Sobel(image, cv2.CV_64F, 1, 1, ksize=3)


def affine(image):
    angle = 90
    scale = 0.5
    rows, cols = image.shape[:2]
    center = (cols // 2, rows // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    return cv2.warpAffine(image, matrix, (cols, rows))


def find_template(image, template):
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(result)
    top_left = max_loc
    h, w = template.shape[:2]
    output = image.copy()
    cv2.rectangle(output, top_left, (top_left[0] + w, top_left[1] + h), (0, 255, 0), 2)
    return output


def negative(image):
    return cv2.bitwise_not(image)


def main():
    image = read_image("123.jpg")
    gray_image = convert_to_gray(image)
    thresh = threshold_image(gray_image, 150)
    filtered = my_filter(image)
    edge = find_edges(gray_image)
    rotated = affine(image)
    neg = negative(image)

    template = read_image("temp.jpg")
    find = find_template(image, template)

    show("Original", image)
    show("Gray", gray_image)
    show("Thresh", thresh)
    show("Filtered", filtered)
    show("Edges", edge)
    show("Affine", rotated)
    show("Negative Satoru", neg)
    show("Found template", find)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
