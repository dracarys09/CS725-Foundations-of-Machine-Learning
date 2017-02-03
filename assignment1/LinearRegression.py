import argparse

parser = argparse.ArgumentParser(description=("Implementation of Linear"
                                              "Regression using Gradient"
                                              "Descent algorithm."))

parser.add_argument("dataset_dir", help="Path of directory containing dataset")
args = parser.parse_args()
