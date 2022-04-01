from settings import Settings
from gradient_boost_model import GradientBoostModel
from fashion_mnist_master.utils import mnist_reader
import data.read_data as rd
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import numpy as np
from os import path


def plot_results(x_value, y1_value, y2_value=None, x_label="", y_axis_label="", y1_legend="", y2_legend="",
                 integer_ticks=False, plot_name=""):
    plt.style.use('ggplot')
    # ['fivethirtyeight', 'seaborn-pastel', 'seaborn-whitegrid', 'ggplot', 'grayscale']

    fig, ax = plt.subplots()

    ax.plot(x_value, y1_value, linewidth=2.0, label=y1_legend, color="blue")
    if y2_value is not None:
        ax.plot(x_value, y2_value, linewidth=2.0, label=y2_legend, color="orange")
        ax.legend()

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_axis_label)
    if integer_ticks:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # ax.set(xlim=(0, 8), xticks=np.arange(1, 8), ylim=(0, 8), yticks=np.arange(1, 8))

    i = 0
    while path.exists("plots/" + plot_name + str(i) + ".png"):
        i += 1

    plt.savefig("plots/" + plot_name + str(i))

    plt.show()


def test_gradient_boosting(x_train, y_train, x_test, y_test, settings):
    gradient_boost_model = GradientBoostModel(settings)

    gradient_boost_model.train(x_train, y_train)

    # results = gradient_boost_model.evaluate(x_test, y_test)
    error_rate_train = gradient_boost_model.error_rate(x_train, y_train)
    error_rate_test = gradient_boost_model.error_rate(x_test, y_test)

    accuracy_train = 1 - error_rate_train
    accuracy_test = 1 - error_rate_test

    return accuracy_train, accuracy_test


def test_and_plot(x_train, y_train, x_test, y_test,
                  n_estimators=None,
                  learning_rate=None,
                  max_depth=None,
                  subsample=None,
                  random_state=None):
    if subsample is None:
        subsample = []
    if n_estimators is None:
        n_estimators = []
    if learning_rate is None:
        learning_rate = []
    if random_state is None:
        random_state = []
    if max_depth is None:
        max_depth = []

    accuracy_train = []
    accuracy_test = []
    for n in n_estimators:
        settings = Settings(n_estimators=n)
        accuracy_train_value, accuracy_test_value = test_gradient_boosting(x_train, y_train, x_test, y_test, settings)
        accuracy_train.append(accuracy_train_value)
        accuracy_test.append(accuracy_test_value)

    plot_results(n_estimators, accuracy_train, accuracy_test, x_label="number of estimators", plot_name="n_estimators",
                 integer_ticks=True, y1_legend="train accuracy", y2_legend="validation accuracy",
                 y_axis_label="accuracy")
    print("number of estimators\n", n_estimators)
    print("train accuracy\n", accuracy_train)
    print("validation accuracy\n", accuracy_test)

    accuracy_train = []
    accuracy_test = []
    for depth in max_depth:
        settings = Settings(max_depth=depth)
        accuracy_train_value, accuracy_test_value = test_gradient_boosting(x_train, y_train, x_test, y_test, settings)
        accuracy_train.append(accuracy_train_value)
        accuracy_test.append(accuracy_test_value)

    plot_results(max_depth, accuracy_train, accuracy_test, x_label="max depth", plot_name="max_depth",
                 integer_ticks=True, y1_legend="train accuracy", y2_legend="validation accuracy",
                 y_axis_label="accuracy")
    print("max depth\n", max_depth)
    print("train accuracy\n", accuracy_train)
    print("validation accuracy\n", accuracy_test)

    accuracy_train = []
    accuracy_test = []
    for rate in learning_rate:
        settings = Settings(learning_rate=rate)
        accuracy_train_value, accuracy_test_value = test_gradient_boosting(x_train, y_train, x_test, y_test, settings)
        accuracy_train.append(accuracy_train_value)
        accuracy_test.append(accuracy_test_value)

    plot_results(learning_rate, accuracy_train, accuracy_test, x_label="learning rate", plot_name="learning_rate",
                 y1_legend="train accuracy", y2_legend="validation accuracy", y_axis_label="accuracy")
    print("learning rate\n", learning_rate)
    print("train accuracy\n", accuracy_train)
    print("validation accuracy\n", accuracy_test)

    accuracy_train = []
    accuracy_test = []
    for portion in subsample:
        settings = Settings(subsample=portion)
        accuracy_train_value, accuracy_test_value = test_gradient_boosting(x_train, y_train, x_test, y_test, settings)
        accuracy_train.append(accuracy_train_value)
        accuracy_test.append(accuracy_test_value)

    plot_results(subsample, accuracy_train, accuracy_test, x_label="subsample portion", plot_name="subsample_portion",
                 y1_legend="train accuracy", y2_legend="validation accuracy", y_axis_label="accuracy")
    print("subsample portion\n", subsample)
    print("train accuracy\n", accuracy_train)
    print("validation accuracy\n", accuracy_test)


def main():
    x_train, y_train, x_test, y_test, x_final_test, y_final_test = rd.read_data()

    just_one_run(x_train, y_train, x_final_test, y_final_test,
                 n_estimators=100,
                 learning_rate=0.4,
                 max_depth=3,
                 subsample=0.5)

    n_estimators_vector = np.array([1, 10, 20, 40, 60, 80, 100])
    n_estimators_vector = np.array([])

    learning_rate_vector = [0.1, 0.3, 0.5, 0.7, 0.9]
    learning_rate_vector = []

    max_depth_vector = np.array([1, 2, 3, 4])
    max_depth_vector = np.array([])

    subsample_vector = np.array([0.1, 0.3, 0.5, 0.7, 0.9, 1.0])

    test_and_plot(x_train, y_train, x_test, y_test,
                  n_estimators=n_estimators_vector,
                  learning_rate=learning_rate_vector,
                  max_depth=max_depth_vector,
                  random_state=None,
                  subsample=subsample_vector)


def just_one_run(x_train, y_train, x_test, y_test,
                 n_estimators=None,
                 learning_rate=None,
                 max_depth=None,
                 subsample=None,
                 random_state=None):
    settings = Settings(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
                        subsample=subsample)
    accuracy_train, accuracy_test = test_gradient_boosting(x_train, y_train, x_test, y_test, settings)

    plot_results(n_estimators, accuracy_train, accuracy_test, x_label="number of estimators",
                 plot_name="n_estimators",
                 integer_ticks=True, y1_legend="train accuracy", y2_legend="validation accuracy",
                 y_axis_label="accuracy")
    print("number of estimators\n", n_estimators)
    print("learning_rate\n", learning_rate)
    print("max_depth\n", max_depth)
    print("number of estimators\n", n_estimators)
    print("subsample\n", subsample)
    print("train accuracy\n", accuracy_train)
    print("test accuracy\n", accuracy_test)


if __name__ == '__main__':
    main()
