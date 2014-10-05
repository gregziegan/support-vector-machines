import argparse
import scipy.io

DATA_DIRECTORY = '../data/'


def solve_ssvm(data):
    print(data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SSVM Solver")
    parser.add_argument('data_file_name')
    args = parser.parse_args()
    filepath = DATA_DIRECTORY + args.data_file_name
    data = scipy.io.loadmat(filepath)
    data_key = args.data_file_name.replace('.mat', '')
    solve_ssvm(data[data_key])