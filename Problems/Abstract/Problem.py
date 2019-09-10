import abc


class Problem(abc.ABCMeta('ABC', (object,), {'__slots__': ()})):
    'Common base class for all problems'
    __name = None  # Path to dataset
    __data_path = None
    __data = None

    def __init__(self, name, data_path):
        'Constructor'
        self.__name = name
        self.__data_path = data_path
        self.set_data(self.read_data())

    # Getters

    def get_name(self):
        return self.__name

    def get_data_path(self):
        return self.__data_path

    def get_data(self):
        return self.__data

    def get_classes(self):
        'Return a list with all classes'
        return list(set(self.__data['Y']))

    # Setters

    def set_data(self, data):
        'Retrieves specified data'
        self.__data = data

    # Other methods

    def preprocess_image(self, image):
        'Preprocess images for contrast enhacement'

        # Dynamic Range shrinking
        #
        # from skimage.exposure import rescale_intensity
        # image = rescale_intensity(image, in_range='image', out_range=(0,255))
        #
        # Adaptative histogram equalization
        #
        # import cv2
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(7,7))
        # image = clahe.apply(image)
        #
        # Global histogram equalization
        #
        # import cv2
        # image = cv2.equalizeHist(image)

        return image

    @abc.abstractmethod
    def read_data(self):
        pass
