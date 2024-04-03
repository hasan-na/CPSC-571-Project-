import tensorflow as tf
import pandas as pd
import numpy as np
import os


def main():
    data = pd.read_csv(os.path.join('data', 'train.csv'))
    pd.set_option('display.max_columns', None)
    print(data.head())


if __name__ == '__main__':
    main()