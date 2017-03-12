import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import MinMaxScaler

file_training_path = 'data.csv'
file_testing_path = 'test.csv'


def preprocess(data):
    data = data.drop(['color'], axis=1)
    data['gender'] -= 1
    data['injury'] -= 1
    data['race'] = data['race'].astype('category', categories=[1, 4, 5, 6, 8, 9])

    data = pd.get_dummies(data, columns=["race"])
    min_max = MinMaxScaler()
    scaled = min_max.fit_transform(data)
    data = pd.DataFrame(scaled, index=data.index, columns=data.columns)
    return data


def load_training_data():
    df = pd.read_csv(file_training_path)
    df.drop('id', axis=1, inplace=True)
    return preprocess(df.drop(['jump'], axis=1)), df[['jump']]


def load_testing_data():
    df = pd.read_csv(file_testing_path)
    df.drop('id', axis=1, inplace=True)
    return preprocess(df)


def main():
    training_X, training_y = load_training_data()
    testing_X = load_testing_data()
    cv = LassoCV(cv=4, max_iter=100000, normalize=True).fit(training_X, training_y.values.ravel())

    lasso = Lasso(alpha=cv.alpha_, normalize=False)
    lasso.fit(training_X, training_y.values.ravel())
    prediction = lasso.predict(testing_X)
    print prediction


if __name__ == '__main__':
    main()
