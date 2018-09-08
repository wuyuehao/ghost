from . import ImageClassificationKeras as ick

class HyperParameters:
    def __init__(self):
        self.learning_rate = 0.01
        self.size_of_hidden_layer = 1

    def runAndGetError(self):
        loss = ick.train(num_layers=self.size_of_hidden_layer, learning_rate=self.learning_rate)
        return loss

hyperparameters = HyperParameters();


def main():
    loss = ick.train()
    return loss


if __name__ == "__main__":
    main()
