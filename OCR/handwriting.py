from sklearn import datasets
from sklearn import model_selection
from sklearn import svm
from sklearn.metrics import accuracy_score


def main():
    data = datasets.load_digits()
    features = data['data']
    labels = data['target']

    accuracy = 0
    for iteration in range(100):
        iteration += 1
        print(iteration)
        train_X, validation_X, train_y, validation_y = model_selection.train_test_split(features, labels,
                                                                                        test_size=0.25)
        clf = svm.SVC(kernel='poly')
        clf.fit(train_X, train_y)
        predictions = clf.predict(validation_X)
        accuracy += accuracy_score(validation_y, predictions)

    print(accuracy / 100)


if __name__ == '__main__':
    main()
