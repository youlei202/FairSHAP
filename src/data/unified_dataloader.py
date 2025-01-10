import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import copy
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def load_dataset(dataset_name):
    """
    加载指定的数据集
    
    参数:
    dataset_name (str): 数据集名称 ('german_credit' 或 'uci')
    
    返回:
    orginal_data (pd.DataFrame): 原始数据集
    processed_data (pd.DataFrame): 预处理后的数据集
    """
    if dataset_name == 'german_credit':
        # load German Credit dataset
        original_data, processed_data = german_credit()
        
    elif dataset_name == 'uci':
        # load UCI dataset
        original_data, processed_data = uci()

    else:
        raise ValueError(
        f"Unknown dataset: {dataset_name}. If you want to use the German Credit Dataset, please input 'german_credit'. If you want to use the UCI Dataset, please input 'uci'."
    )
    
    return original_data, processed_data



def german_credit():
    '''Load the German Credit dataset and preprocess it.'''

    # 加载数据
    data = pd.read_csv('dataset/german_credit/german_credit.csv')
    original_data = data.copy()

    # 打印每列的缺失值数量
    missing_values = data.isnull().sum()
    # print(missing_values)

    # 处理缺失值
    saving_imputer = SimpleImputer(strategy='most_frequent')
    checking_imputer = SimpleImputer(strategy='most_frequent')
    data['Saving accounts'] = saving_imputer.fit_transform(data[['Saving accounts']]).ravel()
    data['Checking account'] = checking_imputer.fit_transform(data[['Checking account']]).ravel()

    # 1. 标准化连续变量
    scaler = StandardScaler()
    data[['Age', 'Credit amount', 'Duration']] = scaler.fit_transform(data[['Age', 'Credit amount', 'Duration']])

    # 2. 性别变量编码
    data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})

    # 3. 分类变量独热编码
    categorical_columns = ['Job', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']
    data = pd.get_dummies(data, columns=categorical_columns, drop_first=False)

    # 将独热编码的布尔值转为数值
    for column in data.select_dtypes(include=['bool']).columns:
        data[column] = data[column].astype(int)

    # 4. 目标变量映射
    data['Risk'] = data['Risk'].map({'good': 0, 'bad': 1})

    # 将 Risk 列移动到最后一列
    risk_column = data.pop('Risk')
    data['Risk'] = risk_column

    # 重命名性别列
    data.rename(columns={'Sex': 'sex'}, inplace=True)

    return original_data, data


def uci():
    
    '''Load the UCI dataset and preprocess it.'''

    # column names
    column_names = [
        "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
        "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
        "hours-per-week", "native-country", "income"
    ]

    # import dataset from adult.data
    original_data = pd.read_csv('dataset/uci_dataset/adult.data', header=None, names=column_names, skipinitialspace=True)
    df = original_data.copy()
    df = df.dropna()

    # 0 represents <=50K, 1 represents >50K
    df['income'] = df['income'].map({'<=50K': 0, '>50K': 1})
    # 0 represents Female, 1 represents Male
    label_encoder_sex = LabelEncoder()
    df['sex'] = label_encoder_sex.fit_transform(df['sex'])

    # 数值和类别特征列
    numeric_features = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    categorical_features = ['workclass', 'education', 'marital-status', 'occupation', 'relationship','race','native-country']

    # numeric_features --- Scaler
    scaler = StandardScaler()
    df_numeric = scaler.fit_transform(df[numeric_features])
    df_numeric = pd.DataFrame(df_numeric, columns=numeric_features)  # , index=X_labeled.index  means keep the same index

    # categorical_features --- OneHot encoder
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    df_categorical = encoder.fit_transform(df[categorical_features])
    # get feature names after one-hot encoding
    encoded_feature_names = encoder.get_feature_names_out(categorical_features)
    df_categorical = pd.DataFrame(df_categorical, columns=encoded_feature_names)

    # sensitive features 
    sex_column = df['sex']
    # race_column = df['race']

    # target feature
    target = df['income']

    # df_processed = pd.concat((df_numeric, sex_column, race_column, df_categorical, target), axis=1)

    df_processed = pd.concat((df_numeric, sex_column, df_categorical, target), axis=1)
    # print(f'X_processed shape: {df_processed.shape}')
    return original_data, df_processed

if __name__ == '__main__':
    a, b =german_credit()
    print(a)
    print(b)

    # c, d = uci()
    # print(c)
    # print(d)