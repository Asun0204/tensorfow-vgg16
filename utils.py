import tflearn.datasets.oxflower17 as oxflower17

def get_oxflower17_data(num_train=1000, num_validation=180, num_test=180):
    # Load the raw oxflower17 data
    X, y = oxflower17.load_data()

    # Shuffle the data

    # Subsample the data

    return X_train, y_train, X_val, y_val, X_test, y_test