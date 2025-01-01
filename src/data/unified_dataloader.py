import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import copy

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

    original_data = pd.read_csv('dataset/german_credit/german_credit.csv')
    processed_data = original_data.copy()
    target = 'Risk'
    # encode categorical variables, Risk is encoded as 1 (bad) and 0 (good)
    label_encoders = {}
    for column in processed_data.select_dtypes(include=['object']).columns:
        if column != target:
            label_encoders[column] = LabelEncoder()
            processed_data[column] = label_encoders[column].fit_transform(processed_data[column])
        else:
            processed_data[target] = processed_data[target].map({'good': 0, 'bad': 1})
    return original_data, processed_data

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
    # a, b =german_credit()
    # print(a)
    # print(b)

    c, d = uci()
    print(c)
    print(d)