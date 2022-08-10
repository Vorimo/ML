from sklearn.datasets import make_moons
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.tree import DecisionTreeClassifier, export_graphviz

if __name__ == '__main__':
    moons_data = make_moons(n_samples=10000, noise=1.4)
    train_x, test_x = train_test_split(moons_data[0], test_size=0.2, random_state=42)
    train_y, test_y = train_test_split(moons_data[1], test_size=0.2, random_state=42)
    tree_classifier = DecisionTreeClassifier(max_leaf_nodes=17, max_features=2, max_depth=8)
    tree_classifier.fit(train_x, train_y)
    final_predictions = tree_classifier.predict(test_x)
    print('Final predictions:', final_predictions)
    clean_cross_predictions = cross_val_predict(tree_classifier, test_x, test_y, cv=3)
    print('F1 score:', f1_score(test_y, clean_cross_predictions, average='macro'))  # ~64%

    # visualization
    export_graphviz(
        tree_classifier,
        out_file='tree_graphs/moons_tree.dot',
        rounded=True,
        filled=True
    )
