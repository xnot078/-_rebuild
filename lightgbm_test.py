from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import joblib


# 載入資料
iris = load_iris()
data = iris.data
target = iris.target


# 劃分訓練資料和測試資料
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)


# 模型訓練
gbm = LGBMClassifier(num_leaves=31, learning_rate=0.05, n_estimators=20)
gbm.fit(X_train, y_train, eval_set=[(X_test, y_test)])#, early_stopping_rounds=5)


# 模型儲存
joblib.dump(gbm, 'loan_model.pkl')
# 模型載入
gbm = joblib.load('loan_model.pkl')


# 模型預測
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)


# 模型評估
print('The accuracy of prediction is:', accuracy_score(y_test, y_pred))


# 特徵重要度
print('Feature importances:', list(gbm.feature_importances_))


# 網格搜尋，引數優化
estimator = LGBMClassifier(num_leaves=31)
param_grid = {
    'learning_rate': [0.01, 0.1, 1],
    'n_estimators': [20, 40]
}
gbm = GridSearchCV(estimator, param_grid)
gbm.fit(X_train, y_train)
print('Best parameters found by grid search are:', gbm.best_params_)
