import numpy as np
import pandas as pd
from factor_analyzer import FactorAnalyzer
from matplotlib import pyplot as plt
from scipy import stats
from scipy.cluster import hierarchy
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from ydata_profiling import ProfileReport
import seaborn as sns



class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, features):
        self.features = features

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X = X[self.features]
        return X


class BoxCoxTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, feature_name, lambda_=1):
        self.feature_name = feature_name
        self.lambda_ = lambda_

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()

        X[self.feature_name] = stats.boxcox(X[self.feature_name].to_numpy(), lmbda=self.lambda_)
        return X


class FA(BaseEstimator, TransformerMixin):
    def __init__(self, n_factors=6, rotation=None):
        self.n_factors = n_factors
        self.rotation = rotation

    def fit(self, X, y=None):
        self.fa = FactorAnalyzer(n_factors=self.n_factors, rotation=self.rotation)
        self.fa.fit(X)
        return self

    def transform(self, X):
        return self.fa.transform(X)


def evaluate(grid, X_test, y_test, df_test):
    # Show the best model
    print('Best model: ', grid.best_estimator_)

    # Show the best parameters
    print('Best R2 grid: ', grid.best_score_)

    # Show the best R2 score
    print('Best R2 estimator: ', grid.best_estimator_.score(X_test, y_test))

    # Show the best RMSE
    y_pred = grid.best_estimator_.predict(X_test)

    print('Best RMSE estimator: ', np.sqrt(mean_squared_error(y_test, y_pred)))

    # Create a predicted vs true plot
    plt.figure(figsize=(10, 10))
    plt.scatter(y_test, y_pred.flatten(), c='crimson')
    p1 = max(max(y_pred.flatten()), max(y_test))
    p2 = min(min(y_pred.flatten()), min(y_test))
    plt.plot([p1, p2], [p1, p2], 'b-')
    plt.xlabel('True Values', fontsize=15)
    plt.ylabel('Predictions', fontsize=15)
    plt.axis('equal')
    plt.show()

    plt.figure(figsize=(25, 7))
    plt.plot(y_test.values, label='True')
    plt.plot(y_pred.flatten(), label='Predicted')
    plt.legend()
    plt.show()

    # test_df = df_test[df_test.index > X_test.index.min()]
    # y_pred = grid.predict(test_df)
    # test_df['a_BAL_SEPA_Q_Y_pred'] = y_pred
    # test_df = test_df[test_df['a_BAL_SEPA_Q_Y'] != 0]
    # test_df = test_df.reset_index(drop=True)
    # plt.figure(figsize=(20, 10))
    # plt.plot(test_df['a_BAL_SEPA_Q_Y_pred'], label='Pred')
    # plt.plot(test_df['a_BAL_SEPA_Q_Y'].fillna(method='ffill'), label='True', c='r')
    # plt.legend()
    # plt.show()


def rfr_feature_importance(X, y, top_n=100):
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    importances = rf.feature_importances_

    feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
    feature_importances = feature_importances.sort_values('Importance', ascending=False)
    feature_importances = feature_importances.head(top_n)

    # Plot the feature importances
    plt.bar(feature_importances['Feature'], feature_importances['Importance'])
    plt.xticks(rotation=90)
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.title('Feature Importances')
    plt.tight_layout()
    plt.show()

    return feature_importances


def sql_tag_inputs(path: str):
    df = pd.read_csv(path, dtype={'Tag Name': str}, sep=';')
    tags = df['Tag Name']
    quoted_tags = ','.join("'" + tags.astype(str) + "'")
    bracket_tags = ','.join("[" + tags.astype(str) + "]")
    return quoted_tags, bracket_tags


def eda(df, name="your_report.html", minimal=True):
    profile = ProfileReport(df, title="Profiling Report", minimal=minimal)
    profile.to_file(name)


def corr_plot(df):
    corr_matrix = df.corr()

    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    plt.figure(figsize=(10, 10))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm')
    plt.tight_layout()
    plt.show()


def scatter_plot(df, target):
    columns = df.columns.tolist()
    columns.remove(target)

    num_plots = len(columns)
    num_rows = int(num_plots / 3) + (num_plots % 3 > 0)
    num_cols = 3

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
    fig.tight_layout(pad=3.0)

    for i, col in enumerate(columns):
        row = i // num_cols
        col = i % num_cols

        ax = axes[row, col] if num_rows > 1 else axes[col]
        sns.scatterplot(x=columns[i], y=target, data=df, ax=ax)
        ax.set_xlabel(columns[i])
        ax.set_ylabel(target)

    plt.show()


def ccf_values(series1, series2):
    p = series1
    q = series2
    p = (p - np.mean(p)) / (np.std(p) * len(p))
    q = (q - np.mean(q)) / (np.std(q))
    c = np.correlate(p, q, 'full')
    return c


def ccf_plot(lags, ccf):
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(lags, ccf)
    ax.axhline(-2 / np.sqrt(23), color='red', label='5% confidence interval')
    ax.axhline(2 / np.sqrt(23), color='red')
    ax.axvline(x=0, color='black', lw=1)
    ax.axhline(y=0, color='black', lw=1)
    ax.axhline(y=np.max(ccf), color='blue', lw=1, linestyle='--', label='highest +/- correlation')
    ax.axhline(y=np.min(ccf), color='blue', lw=1, linestyle='--')
    ax.set(ylim=[-1, 1])
    ax.set_title('Cross Correlation IElTS Search and Registration Count', weight='bold', fontsize=15)
    ax.set_ylabel('Correlation Coefficients', weight='bold', fontsize=12)
    ax.set_xlabel('Time Lags', weight='bold', fontsize=12)
    plt.legend()

# TODO: Granger Causality Test

def hierarchy_dendrogram(X):
    plt.figure(figsize=(10, 7))
    plt.title("Dendogram")
    dend = hierarchy.dendrogram(hierarchy.linkage(X, method='ward'))
    # plt.axhline(65)
    plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    plt.xlabel('Observations')
    plt.ylabel('Height')
    # plt.savefig("plt_dendrogram.png")
    plt.show()


def pca_plots(X):
    data = X.copy()
    pca = PCA()
    scaler = StandardScaler()
    X = scaler.fit(data)
    pca.fit(data)
    # Calculate variance and labels
    var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
    labels = ['PC' + str(x) for x in range(1, len(var) + 1)]

    # Create subplots
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))

    # Plot scree plot
    axs[0].bar(x=range(1, len(var) + 1), height=var, tick_label=labels)
    axs[0].set_ylabel('Variance')
    axs[0].set_xlabel('Principal Component')
    axs[0].set_title('Scree Plot')

    # Plot elbow plot
    axs[1].plot(np.cumsum(pca.explained_variance_ratio_))
    axs[1].set_xlabel('Number of Components')
    axs[1].set_ylabel('Variance (%)')
    axs[1].set_title('Elbow Plot')

    # Adjust spacing between subplots
    plt.subplots_adjust(wspace=0.4)

    # Display the subplots
    plt.show()