import joblib
import pandas as pd
from sklearn.model_selection import GridSearchCV
from skopt import BayesSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from skopt.space import Real, Categorical, Integer
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


class Model:
    def __init__(self, model_name, dataset_path):
        self.grid = None
        self.y_train = None
        self.y_test = None
        self.X_test = None
        self.X_train = None
        self.scaler = None
        self.Y = None
        self.X = None
        self.model_name = model_name
        self.dataset_path = dataset_path

        self.df = pd.read_csv(self.dataset_path)
        self.df = self.df.drop('Unnamed: 0', axis=1)

    def data_preparation(self, is_normalize=0, is_pca=0, is_toBinaryClass=0):
        self.X, self.Y = self.df.drop('label', axis=1), self.df["label"]

        if is_pca != 0:
            print(f"Data dilakukan PCA sebanyak {is_pca}")
            pca = PCA(n_components=is_pca)
            self.X = pd.DataFrame(data=pca.fit_transform(self.X))

        elif is_normalize == 1:
            print(f"Data dilakukan Normalisasi")
            self.scaler = StandardScaler()
            self.scaler.fit(self.X)
            self.X = self.scaler.transform(self.X)

        if is_toBinaryClass == 1:
            temp = pd.Series()
            print(f"Mengubah kelas 'print' dan 'replay' menjadi kelas 'spoof'")
            for idx, i in enumerate(self.Y):
                if i == 'replay' or i == 'print':
                    temp.loc[idx] = 'spoof'
                else:
                    temp.loc[idx] = 'real'
            self.Y = temp

        print(self.X, self.Y)

    def grid_search(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.Y, train_size=0.8,
                                                                                random_state=42)

        param_grid = {
            'C': Real(1e-6, 1e+6, prior='log-uniform'),
            'gamma': Real(1e-6, 1e+1, prior='log-uniform')
        }

        weight = {'real': 3, 'spoof': 2.4}
        self.grid = BayesSearchCV(
            SVC(probability=True, cache_size=1000, class_weight=weight, kernel="rbf", decision_function_shape='ovr'),
            param_grid, n_iter=50,
            cv=5
        )

        # ftting the model for grid search
        self.grid.fit(self.X_train, self.y_train)

        # print best parameter after tuning
        print(self.grid.best_params_)

        # print how our model looks after hyper-parameter tuning
        print(self.grid.best_estimator_)
        grid_predictions = self.grid.predict(self.X_test)

        # print classification report
        print(classification_report(self.y_test, grid_predictions))

        print(self.grid.score(self.X_test, self.y_test))

    def save_model(self, save_path):
        joblib.dump(self.grid, f'{save_path}\\{self.model_name}.pkl')
        print(f"Model tersimpan di {save_path}\\{self.model_name}.pkl")


if __name__ == '__main__':
    model1 = Model(
        model_name="lbp_model1",
        dataset_path="C:\\Users\\Marcel\\Desktop\\code\\tugas-akhir\\spoofing-detection\\dataset\\"
                     "kumpulan glcm\\lbp_1_glcm24.csv",
    )

    model1.data_preparation(
        is_pca=0,       # Jumlah n pada PCA, jika tidak ingin PCA, isi 0
        is_normalize=1,   # Jika pca !=0, maka ngecek isNormalize
        is_toBinaryClass=1,
    )

    model1.grid_search()

    # model1.save_model(
    #     save_path="C:\\Users\\Marcel\\Desktop\\code\\tugas-akhir\\spoofing-detection\\dataset\\model"
    # )
