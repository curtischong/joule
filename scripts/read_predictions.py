import numpy as np

def main():
    npzfile = np.load('../calan_out/ocp_predictions.npz')

    # List all the arrays in the npz file
    print(npzfile.files)

    # Access each array using the keys from the list
    for array_name in npzfile.files:
        print(array_name)
        print(npzfile[array_name])

if __name__ == '__main__':
    main()