import uproot
import shutil
import argparse

from utils import *

def main(__type, depth, additional_cuts):

    data_folder = root_folder + root_file_name #root_folder is imported from config_script
    root = uproot.open(data_folder) #use uproot to see the tree in in the root root_file_name

    #read the names sample by sample
    if __type == 'bkg':
        train_sample_name = []
        for n in root.keys():
            train_sample_name.append(str(n).split(";")[0].split("'")[1])

    elif __type == 'sig':
        signal_name = []
        for n in root.keys():
            if 'nosys' in str(n).lower():
                signal_name.append(n)
        train_sample_name = [str(n).split(";")[0].split("'")[1] for n in signal_name]
        #train_sample_name = [str(n).split(";")[0] for n in signal_name]
    print('I find these branches {}'. format(train_sample_name))

    if __type == 'bkg':
        if os.path.exists(splitted_numpy_bkg):
            shutil.rmtree(splitted_numpy_bkg)
            os.makedirs(splitted_numpy_bkg)
        else:
            os.makedirs(splitted_numpy_bkg)

        for name in train_sample_name:
            cutflow_bkg(data_folder, name, depth, additional_cuts)

    elif __type == 'sig':
        if os.path.exists(numpy_sig):
            shutil.rmtree(numpy_sig)
            os.makedirs(numpy_sig)
        else:
            os.makedirs(numpy_sig)
        for name in train_sample_name:
            cutflow_sig(data_folder, name, depth, additional_cuts)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='cuts on root file and conversion to numpy data')

    parser.add_argument('--type', nargs="?", type = str, default = 'sig', help='bkg or sig')
    parser.add_argument('--depth', nargs="?", default = 'middle' , help='preselection or middle')
    parser.add_argument('--clean_data', nargs="?", type = bool, default = True,  help=' apply additional cuts: remove all the value > 1000 and < 0')

    args = parser.parse_args()

    __type = args.type
    depth = args.depth
    additional_cuts = args.clean_data

    if __type == 'bkg':
        root_file_name = 'user.eschanet.allTrees_v2_0_2_bkg_NoSys.root'
    elif __type == 'sig':
        root_file_name = 'user.eschanet.allTrees_v2_0_2_signal_1Lbb_skim.root'

    if depth in ['preselection', 'middle']:
        print(depth)
    else:
        print('no depth selection')

    if additional_cuts:
        print(' additional cuts applied: removing all the value > 1000 and < 0')

    main(__type, depth, additional_cuts)
