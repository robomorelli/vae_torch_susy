#!/usr/bin/env python3
from utils import *

def main():

    print('"\x1b[31m\"merging the bkg files "\x1b[0m"')
    path_in = splitted_numpy_bkg
    path_out = numpy_bkg

    if os.path.exists(path_out):
        print('folder already existing...saving')
    else:
        try:
            os.makedirs(path_out)
        except:
            print('failing to create output folder')
    concatenate_file(path_in, path_out, 'background')

if __name__ == "__main__":

    main()