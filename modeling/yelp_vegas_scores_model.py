from inspect import signature

import numpy as np
import pandas as pd
from sklearn import svm, clone
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, \
    precision_recall_curve, mean_squared_error
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from db.mysql import Engine
from sklearn.externals import joblib
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB, BernoulliNB
from sklearn import tree
import graphviz
import matplotlib.pyplot as plt
from inspect import signature
from xgboost.sklearn import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

db_conn = Engine.get_db_conn()
df = pd.read_sql('yelp_vegas_final', con=db_conn)

y = df['score']
X = df.drop(['score', 'stars', 'review_count'], axis=1)

# Applied PCA with 10 components, even though it sped up the running time, it didn't improve the accuracies.
# Also PCA negative values cannot be applied to Multinomial and Complement Naive Bayes

# pca = PCA(n_components=10)
# X = pd.DataFrame(pca.fit_transform(X))
# print(pca.explained_variance_ratio_)
# print(pca.singular_values_)

print(df.shape)
print(df.columns)

X_train_final, X_test_final, y_train_final, y_test_final = train_test_split(X, y, test_size=0.20,
                                                                            random_state=100, stratify=y)


def run_model(model, name, folds=5):
    k_fold = StratifiedKFold(n_splits=folds, random_state=100, shuffle=True)
    final_model = clone(model)
    cross_val_scores_mean = []
    cross_val_scores_std = []
    for train_index, test_index in k_fold.split(X, y):
        X_train, X_test = pd.DataFrame(data=X, index=train_index), pd.DataFrame(data=X, index=test_index)
        y_train, y_test = pd.DataFrame(data=y, index=train_index), pd.DataFrame(data=y, index=test_index)

        model.fit(X_train, y_train.values.ravel())
        scores = cross_val_score(model, X_train, y_train.values.ravel(), cv=5)
        cross_val_scores_mean.append(scores.mean())
        cross_val_scores_std.append(scores.std() * 2)

    print('*************************************** {} ***************************************'.format(name))

    cross_val_scores = np.array([cross_val_scores_mean, cross_val_scores_std])
    print('K-fold with %s splits along with cross_validate accuracy scores %0.2f (+/- %0.2f): ' %
          (str(k_fold), cross_val_scores[0].mean(), cross_val_scores[1].std()))

    final_model.fit(X_train_final, y_train_final)
    scores_final = cross_val_score(final_model, X_train_final, y_train_final, cv=5)
    y_pred_final = final_model.predict(X_test_final)
    print('cross validation mean Accuracy score %0.2f (+/- %0.2f)' % (scores_final.mean(), scores_final.std() * 2))
    print('Train accuracy score:', accuracy_score(y_train_final, model.predict(X_train_final)))
    print('Test accuracy score:', accuracy_score(y_test_final, y_pred_final))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test_final, y_pred_final))
    print("Classification Report:")
    print(classification_report(y_test_final, y_pred_final))
    # ROC curve and  precision_recall_curve can only be applied to binary classifications
    # print("ROC curve:", roc_auc_score(y_test, y_pred))
    # precision, recall, thresholds = precision_recall_curve(y_test, y_pred)
    # plot_precision_recall(precision, recall)


def plot_precision_recall(recall, precision, average_precision=None):
    step_kwargs = (
        {'step': 'post'}
        if 'step' in signature(plt.fill_between).parameters else {}
    )
    plt.step(recall, precision, alpha=0.1, where='post')
    plt.fill_between(recall, precision, alpha=0.1, **step_kwargs)

    plt.xlabel('Recall')
    plt.ylabel('Precision')

    if average_precision is not None:
        plt.title('Precision-Recall curve: Average Precision={0:0.2f}'.format(
            average_precision))


# ******************* Decision Tree Classifier (Entropy) ************************
model = tree.DecisionTreeClassifier(criterion='entropy')
run_model(model, 'Decision Tree')
# dot_data = tree.export_graphviz(model, out_file=None)
# graph = graphviz.Source(dot_data)
# graph.render("yelp_decision_tree")

# ******************* Decision Tree Classifier (Gini) ************************
model = tree.DecisionTreeClassifier()
run_model(model, 'Decision Tree')
# dot_data = tree.export_graphviz(model, out_file=None)
# graph = graphviz.Source(dot_data)
# graph.render("yelp_decision_tree")

# ******************* Gaussian Naive Bayes *************************
run_model(GaussianNB(), 'Gaussian Naive Bayes')

# ******************* Multinomial Naive Bayes ***********************
run_model(MultinomialNB(), 'Multinomial Naive Bayes')

# ******************* Complement Naive Bayes ************************
run_model(ComplementNB(), 'Complement Naive Bayes')

# ******************* Bernoulli Naive Bayes *************************
run_model(BernoulliNB(), 'Bernoulli Naive Bayes')

# ******************* KNN ************************
knn_range = range(3, 12)
for k in knn_range:
    print('Iteration: ' + str(k))
    knn_model = KNeighborsClassifier(n_neighbors=k)
    run_model(knn_model, ' KNN({} neighbors) '.format(k))

# # ******************* SVM(linear) ************************
# svm_model = svm.SVC(kernel='linear')
# run_model(svm_model, ' SVM(Linear) ')
#
# # ******************* SVM(poly) ************************
# svm_model = svm.SVC(kernel='poly')
# run_model(svm_model, ' SVM(poly) ')
#
# # ******************* SVM(rbf) ************************
# svm_model = svm.SVC(kernel='rbf')
# run_model(svm_model, ' SVM(rbf) ')

# ******************* Logistic Regression ************************
log_reg = LogisticRegression()
run_model(log_reg, ' Logistic Regression')

# ******************* Ridge (Regularization) 0.01 ************************
ridge_001 = Ridge(alpha=0.01)
ridge_001.fit(X_train_final, y_train_final)
y_pred = model.predict(X_test_final)
scores = cross_val_score(model, X_train_final, y_train_final, cv=5)
print('***************************************Ridge(0.01)***************************************')
print('cross validation score', scores.mean())
ridge_model_001_train_score = ridge_001.score(X_train_final, y_train_final)
ridge_model_001_test_score = ridge_001.score(X_test_final, y_test_final)

mse = mean_squared_error(y_test_final, y_pred)
rmse = np.math.sqrt(mse)
print('RMSE: {}'.format(rmse))
print('Train Score: {}'.format(ridge_model_001_train_score))
print('Test Score: {}'.format(ridge_model_001_test_score))

# ******************* Ridge Regularization (100) ************************
ridge_100 = Ridge(alpha=100)
ridge_100.fit(X_train_final, y_train_final)
y_pred = model.predict(X_test_final)
scores = cross_val_score(model, X_train_final, y_train_final, cv=5)
print('***************************************Ridge(100)***************************************')
print('cross validation score', scores.mean())
ridge_model_100_train_score = ridge_100.score(X_train_final, y_train_final)
ridge_model_100_test_score = ridge_100.score(X_test_final, y_test_final)

mse = mean_squared_error(y_test_final, y_pred)
rmse = np.math.sqrt(mse)
print('RMSE: {}'.format(rmse))
print('Train Score: {}'.format(ridge_model_100_train_score))
print('Test Score: {}'.format(ridge_model_100_test_score))
#
# ******************* Lasso Regularization (0.01) ************************
lasso_001 = Lasso(alpha=0.01)
lasso_001.fit(X_train_final, y_train_final)
y_pred = model.predict(X_test_final)
scores = cross_val_score(model, X_train_final, y_train_final, cv=5)
print('***************************************Lasso(0.01)***************************************')
print('cross validation score', scores.mean())
lasso_model_001_train_score = lasso_001.score(X_train_final, y_train_final)
lasso_model_001_test_score = lasso_001.score(X_test_final, y_test_final)

mse = mean_squared_error(y_test_final, y_pred)
rmse = np.math.sqrt(mse)
print('RMSE: {}'.format(rmse))
print('Train Score: {}'.format(lasso_model_001_train_score))
print('Test Score: {}'.format(lasso_model_001_test_score))

# ******************* Lasso (0.001) ************************
lasso_0001 = Lasso(alpha=0.001)
lasso_0001.fit(X_train_final, y_train_final)
y_pred = model.predict(X_test_final)
scores = cross_val_score(model, X_train_final, y_train_final, cv=5)
print('***************************************Lasso (0.001)***************************************')
print('cross validation score', scores.mean())
lasso_model_0001_train_score = lasso_0001.score(X_train_final, y_train_final)
lasso_model_0001_test_score = lasso_0001.score(X_test_final, y_test_final)

mse = mean_squared_error(y_test_final, y_pred)
rmse = np.math.sqrt(mse)
print('RMSE: {}'.format(rmse))
print('Train Score: {}'.format(lasso_model_0001_train_score))
print('Test Score: {}'.format(lasso_model_0001_test_score))

# ******************* Random Forest ************************
rfc = RandomForestClassifier()
run_model(rfc, 'Random Forest')

# ******************* Gradient Boost Classifier ************************
gbc = GradientBoostingClassifier()
run_model(gbc, 'Gradient Boost Classifier')

# ******************* XG Boost Classifier ************************
xgb = XGBClassifier()
run_model(xgb, 'XG Boost Classifier')

joblib.dump(log_reg, 'log_reg_score_model.joblib')
joblib.dump(xgb, 'xgb_score_model.joblib')
