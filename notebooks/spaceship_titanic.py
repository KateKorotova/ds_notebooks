import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import PowerTransformer

from catboost import CatBoostClassifier, Pool

data = '../data/spaceship-titanic/'

df = pd.read_csv(f'{data}train.csv')
df_test = pd.read_csv(f'{data}test.csv')

cat_cols = df.select_dtypes(include='object').columns.tolist()
num_cols = df.select_dtypes(include='float64').columns.tolist()
bill_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
luxury_cols = ['Spa',  'VRDeck']


def feature_engineering(df):
    df_copy = df.copy()
    df_copy['group'] = df_copy.apply(lambda row: row['PassengerId'].split('_')[0], axis=1)
    df_copy['num_group'] = df_copy.apply(lambda row: row['PassengerId'].split('_')[1], axis=1)
    df_copy['deck'] = df_copy.apply(lambda row: row['Cabin'].split('/')[0] if pd.notna(row['Cabin']) else row['Cabin'], axis=1)
    df_copy['num_cabin'] = df_copy.apply(lambda row: row['Cabin'].split('/')[1] if pd.notna(row['Cabin']) else row['Cabin'],
                               axis=1)
    df_copy['side_cabin'] = df_copy.apply(lambda row: row['Cabin'].split('/')[2] if pd.notna(row['Cabin']) else row['Cabin'],
                                axis=1)
    df_copy['last_name'] = df_copy.apply(lambda row: row['Name'].split()[1] if pd.notna(row['Name']) else row['Name'], axis=1)
    df_copy['total_spending'] = df_copy.apply(lambda row: row[bill_cols].sum(), axis=1)
    df_copy['total_spending_luxuries'] = df_copy.apply(lambda row: row[luxury_cols].sum(), axis=1)

    temp = df_copy.groupby('group').size().reset_index(name='group_size')
    df_copy = df_copy.merge(temp, how='left', on='group')

    return df_copy


def generic_missing_values(df):
    df_copy = df.copy()

    # # fill with mode
    # mode_cols = ['HomePlanet', 'CryoSleep', 'VIP', 'Destination']
    # for col in mode_cols:
    #     mode_value = df_copy[col].mode()[0]
    #     df_copy[col] = df_copy[col].fillna(mode_value)

    # # numeric with zero
    # zero_cols = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    # for col in zero_cols:
    #     df_copy[col] = df_copy[col].fillna(0)

    # last name with the most frequent in a group
    group_last_name_count = df_copy[df_copy['group_size'] > 1].groupby(['group', 'last_name']).size().reset_index(name='count')
    most_frequent_last_name = group_last_name_count.loc[group_last_name_count.groupby('group')['count'].idxmax()]
    df_copy = df_copy.merge(most_frequent_last_name[['group', 'last_name']], on='group', how='left',
                  suffixes=('', '_most_frequent'))
    df_copy['last_name'] = df_copy['last_name'].fillna(df_copy['last_name_most_frequent'])
    df_copy.drop(columns=['last_name_most_frequent'], inplace=True)

    # last name with Unknown for solo travelers
    df_copy['last_name'] = df_copy['last_name'].fillna('Unknown')

    print(df_copy[df_copy.isna().sum(axis=1) > 0].shape)

    df_copy = df_copy.dropna()
    return df_copy


def missing_values(df):
    df_copy = df.copy()

    # In CryoSleep people can not spend
    filtered_rows = (df_copy['CryoSleep'] == True) & (df_copy[bill_cols].isna().any(axis=1))
    df_copy.loc[filtered_rows, bill_cols] = df_copy.loc[filtered_rows, bill_cols].fillna(0.0)

    # If people spent they would not be in CryoSleep
    cryosleep_false = (df_copy['CryoSleep'].isna()) & (df_copy[bill_cols].gt(0).any(axis=1))
    df_copy.loc[cryosleep_false, 'CryoSleep'] = False

    # People from the same group has the same HomePlanet
    groups_homeplanet = df_copy[['group', 'HomePlanet']].groupby(['group', 'HomePlanet']).nunique().reset_index()
    df_copy = df_copy.merge(groups_homeplanet, on='group', how='left', suffixes=('', '_from_group'))
    df_copy['HomePlanet'] = df_copy['HomePlanet'].fillna(df_copy['HomePlanet_from_group'])
    df_copy.drop(columns=['HomePlanet_from_group'], inplace=True)

    # df_copy = df_copy.dropna()
    df_copy = generic_missing_values(df_copy)

    return df_copy


def transforming(df):
    df_copy = df.copy()
    transformer = PowerTransformer(method='yeo-johnson')
    cols_transform = ['RoomService', 'FoodCourt', 'ShoppingMall', 'total_spending_luxuries']
    for col in cols_transform:
        # df_copy[[col]] = transformer.fit_transform(df_copy[[col]])
        df_copy[[col]] = np.log1p(df_copy[[col]])
    return df_copy


def preprocessing(df):
    df_copy = df.copy()
    df_copy = feature_engineering(df_copy)
    df_copy = missing_values(df_copy)
    df_copy = transforming(df_copy)
    return df_copy


def modeling(df):
    df_copy = df.copy()
    df_copy['target'] = df_copy.apply(lambda row: 0 if row['Transported'] == False else 1, axis=1)
    train_cols = [col for col in df_copy.columns if
                  col not in ['PassengerId', 'Cabin', 'Name', 'Transported', 'target',
                              'num_group', 'total_spending',  'Spa', 'VRDeck', 'group']]
    X_train, X_test, y_train, y_test = train_test_split(df_copy[train_cols], df_copy['target'], test_size=0.2,
                                                        random_state=42)
    cat = [cols for cols in cat_cols if cols in train_cols]
    cat_features = cat + ['last_name', 'deck', 'side_cabin']
    train_data = Pool(X_train, y_train, cat_features=cat_features)
    test_data = Pool(X_test, y_test, cat_features=cat_features)
    model = CatBoostClassifier(verbose=True, random_state=42)
    model.fit(train_data)

    preds_class = model.predict(test_data)
    # preds_proba = model.predict_proba(test_data)
    return y_test, preds_class, model


def evaluation(y_test, preds_class):
    print(classification_report(y_test, preds_class))
    print('\n')
    print(f1_score(y_test, preds_class))


def feature_importance(model):
    feature_importances = model.get_feature_importance()
    feature_names = model.feature_names_

    # Create a DataFrame for easy plotting
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    })

    # Sort the DataFrame by importance
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title('CatBoost Feature Importance')
    plt.gca().invert_yaxis()  # To have the highest importance on top
    plt.show()


def result_test(df, model):
    df_copy = df.copy()
    df_copy = preprocessing(df_copy)
    preds_class = model.predict(df_copy)
    df_copy['Transported'] = preds_class
    df_copy[['PassengerId', 'Transported']].to_csv('../submissions/spaceship_titanic/submission_1.csv',
                                                   index=False)


if __name__ == '__main__':
    df_copy = df.copy()
    df_copy = preprocessing(df_copy)
    y_test, preds_class, model = modeling(df_copy)
    evaluation(y_test, preds_class)
    feature_importance(model)