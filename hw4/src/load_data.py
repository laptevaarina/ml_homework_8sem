import urllib.request
from zipfile import ZipFile


def load_data(url, filename):
    urllib.request.urlretrieve(url, filename)
    with ZipFile(filename, 'r') as zf:
        # zf.printdir() # print zip contents
        zf.extractall('images_evaluation')


if __name__ == "__main__":
    url_evaluation = 'https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip'
    filename_evaluation = 'images_evaluation.zip'
    load_data(url_evaluation, filename_evaluation)

    url_background = 'https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip'
    filename_background = 'images_background.zip'
    load_data(url_background, filename_background)
