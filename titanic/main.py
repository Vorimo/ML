import pandas as pd
from numpy import float32
from keras.models import Sequential
from keras.layers import Dense, TextVectorization
from sklearn.preprocessing import MinMaxScaler

if __name__ == '__main__':
    train_x_df = pd.read_csv('datasets/train.csv')
    train_y_df = train_x_df['Survived']
    train_x_df.drop(['Survived', 'PassengerId', 'Name', 'Ticket', 'Fare', 'Embarked', 'Cabin'], axis=1, inplace=True)
    train_x_df['Age'].fillna(train_x_df['Age'].mean(), inplace=True)
    train_x = train_x_df.to_numpy()
    train_x = MinMaxScaler().fit_transform(train_x)
    train_y = train_y_df.to_numpy(dtype=float32)
    #train_x_df['Age'] = MinMaxScaler().fit_transform([train_x_df["Age"]])
    train_x_df['SibSp'] = train_x_df['SibSp'] / 1.
    train_x_df['Parch'] = train_x_df['Parch'] / 1.
    print(train_x_df.head())
    print(train_y_df.head())

    model = Sequential()
    model.add(Dense(input_shape=(5,), activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])
