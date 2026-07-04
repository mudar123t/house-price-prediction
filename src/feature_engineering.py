def encode_finishing_quality(df):
    """
    Encode 'الإكساء' (finishing quality) as an ordinal number:
    ضعيف (weak) = 0, وسط (medium) = 1, جيد (good) = 2, ممتاز (excellent) = 3
    """
    finishing_map = {
        'ضعيف': 0,
        'وسط': 1,
        'جيد': 2,
        'ممتاز': 3
    }
    df = df.copy()
    df['الإكساء'] = df['الإكساء'].map(finishing_map)

    # Safety check: if any value didn't match the map, it becomes NaN - catch that
    if df['الإكساء'].isnull().any():
        unmapped = df[df['الإكساء'].isnull()]
        print("WARNING: some finishing values didn't match the map:", unmapped.shape[0], "rows")

    return df



def encode_binary_columns(df):
    """
    Map binary yes/no and front/back columns to 0/1.
    """
    df = df.copy()

    facing_map = {'أمامي': 1, 'خلفي': 0}
    yes_no_map = {'نعم': 1, 'لا': 0}

    df['الوجهة'] = df['الوجهة'].map(facing_map)
    df['موقف سيارة'] = df['موقف سيارة'].map(yes_no_map)
    df['وجود مصعد'] = df['وجود مصعد'].map(yes_no_map)

    # Safety check for unmapped values
    for col in ['الوجهة', 'موقف سيارة', 'وجود مصعد']:
        if df[col].isnull().any():
            print(f"WARNING: unmapped values found in {col}:", df[col].isnull().sum())

    return df

import pandas as pd

def one_hot_encode_direction(df):
    """
    One-hot encode 'الاتجاه' (direction), since it has no natural order.
    """
    df = df.copy()
    direction_dummies = pd.get_dummies(df['الاتجاه'], prefix='اتجاه')
    df = pd.concat([df.drop(columns=['الاتجاه']), direction_dummies], axis=1)
    return df


def one_hot_encode_direction(df):
    """
    One-hot encode 'الاتجاه' (direction), since it has no natural order.
    drop_first=True avoids the dummy variable trap (multicollinearity),
    which matters for linear models.
    """
    df = df.copy()
    direction_dummies = pd.get_dummies(df['الاتجاه'], prefix='اتجاه', drop_first=True)
    df = pd.concat([df.drop(columns=['الاتجاه']), direction_dummies], axis=1)
    return df


def engineer_features(df):
    """
    Full feature engineering pipeline. Applies all encodings in order.
    Street ('الشارع') and neighborhood ('الحي') are intentionally left
    untouched here - street needs target encoding fit only on training
    data (to avoid leakage), and neighborhood is currently constant.
    """
    df = encode_finishing_quality(df)
    df = encode_binary_columns(df)
    df = one_hot_encode_direction(df)
    return df


def target_encode_column(X_train, X_test, y_train, column, smoothing=10):
    """
    Target (mean) encode a categorical column using ONLY training data,
    with smoothing toward the global mean so categories with few samples
    don't get an extreme, unreliable value.
    """
    global_mean = y_train.mean()
    stats = y_train.groupby(X_train[column]).agg(['mean', 'count'])

    smoothed_map = (stats['count'] * stats['mean'] + smoothing * global_mean) / (stats['count'] + smoothing)

    X_train = X_train.copy()
    X_test = X_test.copy()

    X_train[column + '_encoded'] = X_train[column].map(smoothed_map)
    # Unseen categories in test set fall back to the global mean
    X_test[column + '_encoded'] = X_test[column].map(smoothed_map).fillna(global_mean)

    X_train = X_train.drop(columns=[column])
    X_test = X_test.drop(columns=[column])

    return X_train, X_test, smoothed_map


