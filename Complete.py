import cv2
import numpy as np

def read_image(filepath):
    """
    Считывает цветное изображение из файла.
    :param filepath: путь к изображению.
    :return: цветное изображение.
    """
    image = cv2.imread(filepath)
    if image is None:
        raise FileNotFoundError(f"Изображение по пути {filepath} не найдено.")
    return image

def convert_to_grayscale(image):
    """
    Конвертирует изображение в черно-белое.
    :param image: исходное изображение.
    :return: черно-белое изображение.
    """
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def threshold_image(image, threshold):
    """
    Выполняет пороговое преобразование изображения.
    :param image: черно-белое изображение.
    :param threshold: значение порога.
    :return: изображение после порогового преобразования.
    """
    _, thresh_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    return thresh_image

def apply_custom_filter(image):
    """
    Применяет собственный фильтр к изображению.
    :param image: исходное изображение.
    :return: фильтрованное изображение.
    """
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])  # Пример фильтра (детектор краев)
    return cv2.filter2D(image, -1, kernel)

def detect_edges(image, method='canny'):
    """
    Выполняет поиск границ на изображении с использованием заданного метода.
    :param image: черно-белое изображение.
    :param method: метод поиска границ ('canny', 'laplacian', 'sobel').
    :return: изображение с границами.
    """
    if method == 'canny':
        return cv2.Canny(image, 100, 200)
    elif method == 'laplacian':
        return cv2.Laplacian(image, cv2.CV_64F)
    elif method == 'sobel':
        return cv2.Sobel(image, cv2.CV_64F, 1, 1, ksize=3)
    else:
        raise ValueError("Неизвестный метод: выберите 'canny', 'laplacian' или 'sobel'.")

def affine_transform(image, angle=45, scale=0.5):
    """
    Выполняет аффинное преобразование изображения (сжатие и поворот).
    :param image: исходное изображение.
    :param angle: угол поворота (в градусах).
    :param scale: коэффициент масштабирования.
    :return: трансформированное изображение.
    """
    rows, cols = image.shape[:2]
    center = (cols // 2, rows // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    return cv2.warpAffine(image, matrix, (cols, rows))

def template_matching(image, template):
    """
    Выполняет поиск шаблона в изображении.
    :param image: изображение, в котором производится поиск.
    :param template: шаблон для поиска.
    :return: изображение с выделенным местом шаблона.
    """
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    _, _, _, max_loc = cv2.minMaxLoc(result)
    top_left = max_loc
    h, w = template.shape[:2]
    output = image.copy()
    cv2.rectangle(output, top_left, (top_left[0] + w, top_left[1] + h), (0, 255, 0), 2)
    return output

def morphology_operations(image):
    """
    Выполняет морфологические операции над изображением.
    :param image: черно-белое изображение.
    :return: изображение после морфологических операций.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

# Пример использования:
if __name__ == "__main__":
    filepath = "123.jpg"
    image = read_image(filepath)
    gray_image = convert_to_grayscale(image)
    thresh_image = threshold_image(gray_image, 127)
    filtered_image = apply_custom_filter(gray_image)
    edges = detect_edges(gray_image, method='canny')
    transformed_image = affine_transform(image)

    # template_path = "template.jpg"
    # template = read_image(template_path)
    # matched_image = template_matching(image, template)

    morphed_image = morphology_operations(thresh_image)

    # Сохраняем результаты для визуализации
    # cv2.imwrite("gray_image.jpg", gray_image)
    # cv2.imwrite("thresh_image.jpg", thresh_image)
    # cv2.imwrite("filtered_image.jpg", filtered_image)
    # cv2.imwrite("edges.jpg", edges)
    # cv2.imwrite("transformed_image.jpg", transformed_image)
    # cv2.imwrite("matched_image.jpg", matched_image)
    cv2.imshow("morphed_image.jpg", morphed_image)

    cv2.waitKey(0)
