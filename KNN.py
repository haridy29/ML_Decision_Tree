import math
import pandas


def calculate_distance(point_1, point_2):
    distance = 0
    for i in range(len(point_1)):
        distance += pow(point_1[i] - point_2[i], 2)
    return math.sqrt(distance)


class KNN:
    def __init__(self, csv_file, k):
        self.training_data = None
        self.testing_data = None
        self.training_data_input = None
        self.training_data_output = None
        self.testing_data_input = None
        self.testing_data_output = None
        self.k = k

        self.dataset = pandas.read_csv(csv_file)
        self.shuffle_dataset()
        self.split_into_training_and_testing()
        self.normalize_features()
        self.convert_to_numpy()

    def shuffle_dataset(self):
        self.dataset = self.dataset.sample(frac=1.0)

    def normalize_features(self):
        self.training_data_input = (self.training_data_input - self.training_data_input.mean()) / self.training_data_input.std()
        self.testing_data_input = (self.testing_data_input - self.testing_data_input.mean()) / self.testing_data_input.std()

    def split_into_training_and_testing(self):
        training_data_size = len(self.dataset) * 70 // 100
        self.training_data = self.dataset[:training_data_size]
        self.testing_data = self.dataset[training_data_size:]
        self.training_data_input = self.training_data.drop(['class'], axis=1)
        self.training_data_output = self.training_data['class']
        self.testing_data_input = self.testing_data.drop(['class'], axis=1)
        self.testing_data_output = self.testing_data['class']

    def convert_to_numpy(self):
        self.training_data_input = self.training_data_input.to_numpy()
        self.training_data_output = self.training_data_output.to_numpy()
        self.testing_data_input = self.testing_data_input.to_numpy()
        self.testing_data_output = self.testing_data_output.to_numpy()

    def classify(self, x):
        results = []  # holds pairs of distance and class
        for i in range(len(self.training_data)):
            results.append([calculate_distance(self.training_data_input[i], x), self.training_data_output[i]])
        results.sort()
        count = [0, 0]
        for i in range(self.k):
            count[results[i][1]] += 1
        # print(results)
        # print(count)
        if count[0] == count[1]:
            return results[0][1]
        return 0 if count[0] > count[1] else 1

    def test(self):
        correct = 0
        for i in range(len(self.testing_data)):
            actual = self.testing_data_output[i]
            predicted = self.classify(self.testing_data_input[i])
            if predicted == actual:
                correct += 1

        return correct, len(self.testing_data)


if __name__ == '__main__':
    model = KNN('BankNote_Authentication.csv', 9)
    correct, total = model.test()
    print(f'k value: {model.k}')
    print(f'Number of correctly classified instances: {correct} Total number of instances: {total}')
    print(f'Accuracy = {correct / total}')
