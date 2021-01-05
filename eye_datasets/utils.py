import cv2


def image_to_binimage(image, threshold=10):
    if len(image.shape) > 2:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    ret, binimage = cv2.threshold(gray_image, threshold, 255,
                                  cv2.THRESH_BINARY)
    return binimage


def get_roi(image, area_threshold=10000):
    bin_image = image_to_binimage(image)
    num_labels, labels, stats, centers = \
        cv2.connectedComponentsWithStats(bin_image, connectivity=8,
                                         ltype=cv2.CV_32S)

    roi_list = []
    num_labels_valid = 0
    for t in range(1, num_labels, 1):
        x, y, w, h, area = stats[t]
        if area < area_threshold:
            continue
        num_labels_valid += 1
        roi_list.append((x, y, w, h))

    return num_labels_valid, roi_list


class Resize2D(object):
    def __init__(self, size):
        assert isinstance(size, int)
        self.x = size
        self.y = size

    def __call__(self, image):
        return cv2.resize(image, (self.x, self.y))
