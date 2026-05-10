from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

CLASSIFIERS = {
    'dt': DecisionTreeClassifier(),
    'rf': RandomForestClassifier(n_estimators=100),
    # 'mlp': MLPClassifier(max_iter=1000),
    # 'svm': SVC(probability=True),
    'lr': LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=1000)
}

    #           Bank    Credit  Census  Compas  Meps
    #   DT:     0.8714  0.7750  0.8196  0.8129  0.7907
    #   RF:     0.9025  0.7917  0.8535  0.8299  0.8737
    #   MLP:    0.8964  0.7167  0.8426  0.8284  0.8466
    #   SVM:    0.8844  0.7333  0.8142  0.8266  0.8622
    #   LR:     0.8956  0.7250  0.8131  0.8266  0.8797

# create classifiers
knn_clf = KNeighborsClassifier()
mlp_clf = MLPClassifier()
svm_clf = SVC(probability=True)
rf_clf = RandomForestClassifier()
nb_clf = GaussianNB()


# ensemble above classifiers for majority voting
voting_emseble_classifier = VotingClassifier(
    estimators=[('knn', knn_clf), ('mlp', mlp_clf), ('svm', svm_clf), ('rf', rf_clf), ('nb', nb_clf)],
    voting='soft'
)

# set a pipeline to handle the prediction process
ensemble_pipeline = Pipeline([('scaler', StandardScaler()),
                ('ensemble', voting_emseble_classifier)])


# Bank          0.8978
# Census        0.8429
# Credit        0.7833
# Compas        0.8223
# Meps          0.8708