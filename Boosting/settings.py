class Settings:
    default_n_estimators = 50
    default_learning_rate = 0.1
    default_max_depth = 1
    default_random_state = 0
    subsample = 1.0

    def __init__(self, n_estimators=None,
                 learning_rate=None,
                 max_depth=None,
                 random_state=None,
                 subsample=None
                 ):

        if n_estimators is not None:
            self.n_estimators = n_estimators
        else:
            self.n_estimators = Settings.default_n_estimators

        if learning_rate is not None:
            self.learning_rate = learning_rate
        else:
            self.learning_rate = Settings.default_learning_rate

        if max_depth is not None:
            self.max_depth = max_depth

        else:
            self.max_depth = Settings.default_max_depth
        if random_state is not None:
            self.random_state = random_state
        else:
            self.random_state = Settings.default_random_state

        if subsample is not None:
            self.subsample = subsample
        else:
            self.subsample = Settings.subsample



    def __str__(self):
        my_string = "Model settings:\n"

        my_string = my_string + "n_estimators = " + str(self.n_estimators) + "\n"

        my_string = my_string + "learning_rate = " + str(self.learning_rate) + "\n"

        my_string = my_string + "max_depth = " + str(self.max_depth) + "\n"

        my_string = my_string + "random_state = " + str(self.random_state) + "\n"

        return my_string
