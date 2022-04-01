from fashion_mnist_master.utils import mnist_reader


def read_data():
    portion_of_train_data = 0.8

    x, y = mnist_reader.load_mnist('data/fashion', kind='train')
    number_of_train_data = int((len(y)) * portion_of_train_data)

    x_train, x_test = x[:number_of_train_data], x[number_of_train_data:]
    y_train, y_test = y[:number_of_train_data], y[number_of_train_data:]

    x_final_test, y_final_test = mnist_reader.load_mnist('data/fashion', kind='t10k')
    # return x_train, y_train, x_test, y_test, x_final_test, y_final_test
    return x, y, x_test, y_test, x_final_test, y_final_test
