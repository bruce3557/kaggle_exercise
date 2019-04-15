import tensorflow as tf
import featuretools as ft
import pandas as pd
import math
import os
import sys

from tensorflow.keras import layers
# from sklearn.model_selection import train_tespipt_split

feature_cols = [
    "MSSubClass",
    "MSZoning",
    "LotFrontage",
    "LotArea",
    "Street",
    "Alley",
    "LotShape",
    "LandContour",
    "Utilities",
    "LotConfig",
    "LandSlope",
    "Neighborhood",
    "Condition1",
    "Condition2",
    "BldgType",
    "HouseStyle",
    "OverallQual",
    "OverallCond",
    "YearBuilt",
    "YearRemodAdd",
    "RoofStyle",
    "RoofMatl",
    "Exterior1st",
    "Exterior2nd",
    "MasVnrType",
    "MasVnrArea",
    "ExterQual",
    "ExterCond",
    "Foundation",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinSF1",
    "BsmtFinType2",
    "BsmtFinSF2",
    "BsmtUnfSF",
    "TotalBsmtSF",
    "Heating",
    "HeatingQC",
    "CentralAir",
    "Electrical",
    "1stFlrSF",
    "2ndFlrSF",
    "LowQualFinSF",
    "GrLivArea",
    "BsmtFullBath",
    "BsmtHalfBath",
    "FullBath",
    "HalfBath",
    "BedroomAbvGr",
    "KitchenAbvGr",
    "KitchenQual",
    "TotRmsAbvGrd",
    "Functional",
    "Fireplaces",
    "FireplaceQu",
    "GarageType",
    "GarageYrBlt",
    "GarageFinish",
    "GarageCars",
    "GarageArea",
    "GarageQual",
    "GarageCond",
    "PavedDrive",
    "WoodDeckSF",
    "OpenPorchSF",
    "EnclosedPorch",
    "3SsnPorch",
    "ScreenPorch",
    "PoolArea",
    "PoolQC",
    "Fence",
    "MiscFeature",
    "MiscVal",
    "MoSold",
    "YrSold",
    "SaleType",
    "SaleCondition"
]
target_col = "SalePrice"

origin_data = pd.read_csv(
    sys.argv[1],
    na_values="",
    skipinitialspace=True
)
origin_test_data = pd.read_csv(
    sys.argv[2],
    na_values="",
    skipinitialspace=True
)

def replace_str_nan(x):
    return x if isinstance(x, str) else "?"

def replace_numeric_nan(x):
    return 0 if math.isnan(x) else x

tf.enable_eager_execution()

# print(origin_data[origin_data.isnull().any(axis=1)])
train_target = origin_data[target_col]
train_data = origin_data.drop(columns=[target_col])#.where(pd.notnull(origin_data), "?")
test_data = origin_test_data
test_df = origin_test_data.assign(
    MSZoning=test_data.MSZoning.apply(replace_str_nan),
    LotFrontage=test_data.LotFrontage.apply(replace_numeric_nan),
    Street=test_data.Street.apply(replace_str_nan),
    Alley=test_data.Alley.apply(replace_str_nan),
    MasVnrType=test_data.MasVnrType.apply(replace_str_nan),
    BsmtQual=test_data.BsmtQual.apply(replace_str_nan),
    BsmtCond=test_data.BsmtCond.apply(replace_str_nan),
    BsmtExposure=test_data.BsmtExposure.apply(replace_str_nan),
    BsmtFinType1=test_data.BsmtFinType1.apply(replace_str_nan),
    BsmtFinType2=test_data.BsmtFinType2.apply(replace_str_nan),
    Electrical=test_data.Electrical.apply(replace_str_nan),
    FireplaceQu=test_data.FireplaceQu.apply(replace_str_nan),
    GarageType=test_data.GarageType.apply(replace_str_nan),
    GarageFinish=test_data.GarageFinish.apply(replace_str_nan),
    GarageQual=test_data.GarageQual.apply(replace_str_nan),
    GarageCond=test_data.GarageCond.apply(replace_str_nan),
    PoolQC=test_data.PoolQC.apply(replace_str_nan),
    Fence=test_data.Fence.apply(replace_str_nan),
    MiscFeature=test_data.MiscFeature.apply(replace_str_nan),
    Utilities=test_data.Utilities.apply(replace_str_nan),
    MasVnrArea=test_data.MasVnrArea.apply(replace_numeric_nan),
    Exterior1st=test_data.Exterior1st.apply(replace_str_nan),
    Exterior2nd=test_data.Exterior2nd.apply(replace_str_nan),
    ExterQual=test_data.ExterQual.apply(replace_str_nan),
    ExterCond=test_data.ExterCond.apply(replace_str_nan),
    Foundation=test_data.Foundation.apply(replace_str_nan),
    # BsmtQual=test_data.BsmtQual.apply(replace_str_nan),
    BsmtFinSF1=test_data.BsmtFinSF1.apply(replace_numeric_nan),
    BsmtFinSF2=test_data.BsmtFinSF2.apply(replace_numeric_nan),
    BsmtUnfSF=test_data.BsmtUnfSF.apply(replace_numeric_nan),
    TotalBsmtSF=test_data.TotalBsmtSF.apply(replace_numeric_nan),
    LowQualFinSF=test_data.LowQualFinSF.apply(replace_numeric_nan),
    GrLivArea=test_data.GrLivArea.apply(replace_numeric_nan),
    KitchenQual=test_data.KitchenQual.apply(replace_str_nan),
    Functional=test_data.Functional.apply(replace_str_nan),
    GarageYrBlt=test_data.GarageYrBlt.apply(replace_numeric_nan),
    GarageCars=test_data.GarageCars.apply(replace_numeric_nan),
    GarageArea=test_data.GarageArea.apply(replace_numeric_nan),
    SaleType=test_data.SaleType.apply(replace_str_nan),
    SaleCondition=test_data.SaleCondition.apply(replace_str_nan)
)

test_df["1stFlrSF"] = test_data["1stFlrSF"].apply(replace_numeric_nan)
test_df["2ndFlrSF"] = test_data["2ndFlrSF"].apply(replace_numeric_nan)
    
train_df = train_data.assign(
    MSZoning=train_data.MSZoning.apply(replace_str_nan),
    LotFrontage=train_data.LotFrontage.apply(replace_numeric_nan),
    Street=train_data.Street.apply(replace_str_nan),
    Alley=train_data.Alley.apply(replace_str_nan),
    Utilities=train_data.Utilities.apply(replace_str_nan),
    MasVnrType=train_data.MasVnrType.apply(replace_str_nan),
    BsmtQual=train_data.BsmtQual.apply(replace_str_nan),
    BsmtCond=train_data.BsmtCond.apply(replace_str_nan),
    BsmtExposure=train_data.BsmtExposure.apply(replace_str_nan),
    BsmtFinType1=train_data.BsmtFinType1.apply(replace_str_nan),
    BsmtFinType2=train_data.BsmtFinType2.apply(replace_str_nan),
    Electrical=train_data.Electrical.apply(replace_str_nan),
    FireplaceQu=train_data.FireplaceQu.apply(replace_str_nan),
    GarageType=train_data.GarageType.apply(replace_str_nan),
    GarageFinish=train_data.GarageFinish.apply(replace_str_nan),
    GarageQual=train_data.GarageQual.apply(replace_str_nan),
    GarageCond=train_data.GarageCond.apply(replace_str_nan),
    PoolQC=train_data.PoolQC.apply(replace_str_nan),
    Fence=train_data.Fence.apply(replace_str_nan),
    MiscFeature=train_data.MiscFeature.apply(replace_str_nan),
    MasVnrArea=train_data.MasVnrArea.apply(replace_numeric_nan),
    Exterior1st=train_data.Exterior1st.apply(replace_str_nan),
    Exterior2nd=train_data.Exterior2nd.apply(replace_str_nan),
    ExterQual=train_data.ExterQual.apply(replace_str_nan),
    ExterCond=train_data.ExterCond.apply(replace_str_nan),
    Foundation=train_data.Foundation.apply(replace_str_nan),
    # BsmtQual=train_data.BsmtQual.apply(replace_str_nan)
    BsmtFinSF1=train_data.BsmtFinSF1.apply(replace_numeric_nan),
    BsmtFinSF2=train_data.BsmtFinSF2.apply(replace_numeric_nan),
    BsmtUnfSF=train_data.BsmtUnfSF.apply(replace_numeric_nan),
    TotalBsmtSF=train_data.TotalBsmtSF.apply(replace_numeric_nan),
    # 1stFlrSF=train_data.1stFlrSF.apply(replace_numeric_nan),
    # 2ndFlrSF=train_data.2ndFlrSF.apply(replace_numeric_nan),
    LowQualFinSF=train_data.LowQualFinSF.apply(replace_numeric_nan),
    GrLivArea=train_data.GrLivArea.apply(replace_numeric_nan),
    KitchenQual=train_data.KitchenQual.apply(replace_str_nan),
    Functional=train_data.Functional.apply(replace_str_nan),
    GarageYrBlt=train_data.GarageYrBlt.apply(replace_numeric_nan),
    GarageCars=train_data.GarageCars.apply(replace_numeric_nan),
    GarageArea=train_data.GarageArea.apply(replace_numeric_nan),
    # Fence=train_data.Fence.apply(replace_str_nan),
    # MiscFeature=train_data.MiscFeature.apply(replace_str_nan)
    SaleType=train_data.SaleType.apply(replace_str_nan),
    SaleCondition=train_data.SaleCondition.apply(replace_str_nan)
)
train_df["1stFlrSF"] = train_data["1stFlrSF"].apply(replace_numeric_nan)
train_df["2ndFlrSF"] = train_data["2ndFlrSF"].apply(replace_numeric_nan)
# print(test_df)
# print(dict(train_df))
# print(train_df)

# train_data.dropna()

# print(train_df.head(20))
print(train_df.dtypes)
# print(train_target)
# print(dict(train_data))
# print(train_df.MSSubClass.isnull())

print("\n--------------------------")

def create_train_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices(
        (
            dict(train_df[feature_cols]),
            train_target
        )
    )
    dataset = dataset.shuffle(256).repeat(count=200).batch(64)
    print(dataset)
    return dataset

def create_test_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices(
        (
            dict(test_df[feature_cols])
        )
    )
    dataset = dataset.repeat(count=1).batch(64)
    # dataset = 
    print(dataset)
    return dataset

# ds = tf.data.Dataset.from_tensor_slices(
#     (
#         dict(train_df[feature_cols]),
#         train_target
#     )
# )
# for feature_batch, label_batch in ds.take(1):
#   print('Some feature keys:', list(feature_batch.keys())[:5])
#   print('A batch of RoofStyle  :', feature_batch['RoofStyle'])
#   print('A batch of Labels:', label_batch )

tf_features = [
    tf.feature_column.numeric_column(key="MSSubClass", dtype=tf.int64),
    tf.feature_column.embedding_column(
        dimension=10,
        categorical_column=tf.feature_column.categorical_column_with_vocabulary_list(
            key="MSZoning",
            vocabulary_list=train_df.MSZoning.unique().tolist()
        )
    ),
    tf.feature_column.numeric_column(key="LotFrontage", dtype=tf.float64),
    tf.feature_column.numeric_column(key="LotArea", dtype=tf.int64),
    tf.feature_column.embedding_column(
        dimension=10,
        categorical_column=tf.feature_column.categorical_column_with_vocabulary_list(
            key="Street",
            vocabulary_list=train_df.Street.unique().tolist()
        )
    ),
    tf.feature_column.embedding_column(
        dimension=10,
        categorical_column=tf.feature_column.categorical_column_with_vocabulary_list(
        key="Alley",
        vocabulary_list=train_df.Alley.unique().tolist()
    )),
    tf.feature_column.embedding_column(
        dimension=10,
        categorical_column=tf.feature_column.categorical_column_with_vocabulary_list(
        key="LotShape",
        vocabulary_list=train_df.LotShape.unique().tolist()
    )),
    tf.feature_column.embedding_column(
        dimension=10,
        categorical_column=tf.feature_column.categorical_column_with_vocabulary_list(
        key="LandContour",
        vocabulary_list=train_df.LandContour.unique().tolist()
    )),
    tf.feature_column.embedding_column(
        dimension=10,
        categorical_column=tf.feature_column.categorical_column_with_vocabulary_list(
        key="Utilities",
        vocabulary_list=train_df.Utilities.unique().tolist()
    )),
    tf.feature_column.embedding_column(
        dimension=10,
        categorical_column=tf.feature_column.categorical_column_with_vocabulary_list(
        key="LotConfig",
        vocabulary_list=train_df.LotConfig.unique().tolist()
    )),
    tf.feature_column.embedding_column(
        dimension=10,
        categorical_column=tf.feature_column.categorical_column_with_vocabulary_list(
        key="LandSlope",
        vocabulary_list=train_df.LandSlope.unique().tolist()
    )),
    tf.feature_column.embedding_column(
        dimension=10,
        categorical_column=tf.feature_column.categorical_column_with_vocabulary_list(
        key="Neighborhood",
        vocabulary_list=train_df.Neighborhood.unique().tolist()
    )),
    tf.feature_column.embedding_column(
        dimension=10,
        categorical_column=tf.feature_column.categorical_column_with_vocabulary_list(
        key="Condition1",
        vocabulary_list=train_df.Condition1.unique().tolist()
    )),
    tf.feature_column.embedding_column(
        dimension=10,
        categorical_column=tf.feature_column.categorical_column_with_vocabulary_list(
        key="Condition2",
        vocabulary_list=train_df.Condition2.unique().tolist()
    )),
    tf.feature_column.embedding_column(
        dimension=10,
        categorical_column=tf.feature_column.categorical_column_with_vocabulary_list(
        key="BldgType",
        vocabulary_list=train_df.BldgType.unique().tolist()
    )),
    tf.feature_column.embedding_column(
        dimension=10,
        categorical_column=tf.feature_column.categorical_column_with_vocabulary_list(
        key="HouseStyle",
        vocabulary_list=train_df.HouseStyle.unique().tolist()
    )),
    tf.feature_column.numeric_column(
        key="OverallQual",
        dtype=tf.int64
    ),
    tf.feature_column.numeric_column(
        key="OverallCond",
        dtype=tf.int64
    ),
    tf.feature_column.numeric_column(
        key="YearRemodAdd",
        dtype=tf.int64
    ),
    tf.feature_column.numeric_column(
        key="YearBuilt",
        dtype=tf.int64
    ),
    tf.feature_column.embedding_column(
        dimension=10,
        categorical_column=tf.feature_column.categorical_column_with_vocabulary_list(
        key="RoofStyle",
        vocabulary_list=train_df.RoofStyle.unique().tolist()
    )),
    tf.feature_column.embedding_column(
        dimension=10,
        categorical_column=tf.feature_column.categorical_column_with_vocabulary_list(
        key="RoofMatl",
        vocabulary_list=train_df.RoofMatl.unique().tolist()
    )),
    tf.feature_column.embedding_column(
        dimension=10,
        categorical_column=tf.feature_column.categorical_column_with_vocabulary_list(
        key="Exterior1st",
        vocabulary_list=train_df.Exterior1st.unique().tolist()
    )),
    tf.feature_column.embedding_column(
        dimension=10,
        categorical_column=tf.feature_column.categorical_column_with_vocabulary_list(
        key="Exterior2nd",
        vocabulary_list=train_df.Exterior2nd.unique().tolist()
    )),
    tf.feature_column.embedding_column(
        dimension=10,
        categorical_column=tf.feature_column.categorical_column_with_vocabulary_list(
        key="MasVnrType",
        vocabulary_list=train_df.MasVnrType.unique().tolist()
    )),
    tf.feature_column.numeric_column(
        key="MasVnrArea",
        dtype=tf.float64
    ),
    tf.feature_column.embedding_column(
        dimension=10,
        categorical_column=tf.feature_column.categorical_column_with_vocabulary_list(
        key="ExterQual",
        vocabulary_list=train_df.ExterQual.unique().tolist()
    )),
    tf.feature_column.embedding_column(
        dimension=10,
        categorical_column=tf.feature_column.categorical_column_with_vocabulary_list(
        key="ExterCond",
        vocabulary_list=train_df.ExterCond.unique().tolist()
    )),
    tf.feature_column.embedding_column(
        dimension=10,
        categorical_column=tf.feature_column.categorical_column_with_vocabulary_list(
        key="Foundation",
        vocabulary_list=train_df.Foundation.unique().tolist()
    )),
    tf.feature_column.embedding_column(
        dimension=10,
        categorical_column=tf.feature_column.categorical_column_with_vocabulary_list(
        key="BsmtQual",
        vocabulary_list=train_df.BsmtQual.unique().tolist()
    )),
    tf.feature_column.embedding_column(
        dimension=10,
        categorical_column=tf.feature_column.categorical_column_with_vocabulary_list(
        key="BsmtCond",
        vocabulary_list=train_df.BsmtCond.unique().tolist()
    )),
    tf.feature_column.embedding_column(
        dimension=10,
        categorical_column=tf.feature_column.categorical_column_with_vocabulary_list(
        key="BsmtExposure",
        vocabulary_list=train_df.BsmtExposure.unique().tolist()
    )),
    tf.feature_column.embedding_column(
        dimension=10,
        categorical_column=tf.feature_column.categorical_column_with_vocabulary_list(
        key="BsmtFinType1",
        vocabulary_list=train_df.BsmtFinType1.unique().tolist()
    )),
    tf.feature_column.numeric_column(
        key="BsmtFinSF1",
        dtype=tf.float64
    ),
    tf.feature_column.embedding_column(
        dimension=10,
        categorical_column=tf.feature_column.categorical_column_with_vocabulary_list(
        key="BsmtFinType2",
        vocabulary_list=train_df.BsmtFinType2.unique().tolist()
    )),
    tf.feature_column.numeric_column(
        key="BsmtFinSF2",
        dtype=tf.float64
    ),
    tf.feature_column.numeric_column(
        key="TotalBsmtSF",
        dtype=tf.float64
    ),
    tf.feature_column.embedding_column(
        dimension=10,
        categorical_column=tf.feature_column.categorical_column_with_vocabulary_list(
        key="Heating",
        vocabulary_list=train_df.Heating.unique().tolist()
    )),
    tf.feature_column.embedding_column(
        dimension=10,
        categorical_column=tf.feature_column.categorical_column_with_vocabulary_list(
        key="HeatingQC",
        vocabulary_list=train_df.HeatingQC.unique().tolist()
    )),
    tf.feature_column.embedding_column(
        dimension=10,
        categorical_column=tf.feature_column.categorical_column_with_vocabulary_list(
        key="CentralAir",
        vocabulary_list=train_df.CentralAir.unique().tolist()
    )),
    tf.feature_column.embedding_column(
        dimension=10,
        categorical_column=tf.feature_column.categorical_column_with_vocabulary_list(
        key="Electrical",
        vocabulary_list=train_df.Electrical.unique().tolist()
    )),
    tf.feature_column.numeric_column(
        key="1stFlrSF",
        dtype=tf.float64
    ),
    tf.feature_column.numeric_column(
        key="2ndFlrSF",
        dtype=tf.float64
    ),
    tf.feature_column.numeric_column(
        key="LowQualFinSF",
        dtype=tf.float64
    ),
    tf.feature_column.numeric_column(
        key="GrLivArea",
        dtype=tf.float64
    ),
    tf.feature_column.numeric_column(
        key="BsmtFullBath",
        dtype=tf.int64
    ),
    tf.feature_column.numeric_column(
        key="BsmtHalfBath",
        dtype=tf.int64
    ),
    tf.feature_column.numeric_column(
        key="FullBath",
        dtype=tf.int64
    ),
    tf.feature_column.numeric_column(
        key="HalfBath",
        dtype=tf.int64
    ),
    tf.feature_column.numeric_column(
        key="BedroomAbvGr",
        dtype=tf.int64
    ),
    tf.feature_column.numeric_column(
        key="KitchenAbvGr",
        dtype=tf.int64
    ),
    tf.feature_column.embedding_column(
        dimension=10,
        categorical_column=tf.feature_column.categorical_column_with_vocabulary_list(
        key="KitchenQual",
        vocabulary_list=train_df.KitchenQual.unique().tolist()
    )),
    tf.feature_column.numeric_column(
        key="TotRmsAbvGrd",
        dtype=tf.int64
    ),
    tf.feature_column.embedding_column(
        dimension=10,
        categorical_column=tf.feature_column.categorical_column_with_vocabulary_list(
        key="Functional",
        vocabulary_list=train_df.Functional.unique().tolist()
    )),
    tf.feature_column.numeric_column(
        key="Fireplaces",
        dtype=tf.int64
    ),
    tf.feature_column.embedding_column(
        dimension=10,
        categorical_column=tf.feature_column.categorical_column_with_vocabulary_list(
        key="FireplaceQu",
        vocabulary_list=train_df.FireplaceQu.unique().tolist()
    )),
    tf.feature_column.embedding_column(
        dimension=10,
        categorical_column=tf.feature_column.categorical_column_with_vocabulary_list(
        key="GarageType",
        vocabulary_list=train_df.GarageType.unique().tolist()
    )),
    tf.feature_column.numeric_column(
        key="GarageYrBlt",
        dtype=tf.int64
    ),
    tf.feature_column.embedding_column(
        dimension=10,
        categorical_column=tf.feature_column.categorical_column_with_vocabulary_list(
        key="GarageFinish",
        vocabulary_list=train_df.GarageFinish.unique().tolist()
    )),
    tf.feature_column.numeric_column(
        key="GarageCars",
        dtype=tf.int64
    ),
    tf.feature_column.numeric_column(
        key="GarageArea",
        dtype=tf.int64
    ),
    tf.feature_column.embedding_column(
        dimension=10,
        categorical_column=tf.feature_column.categorical_column_with_vocabulary_list(
        key="GarageQual",
        vocabulary_list=train_df.GarageQual.unique().tolist()
    )),
    tf.feature_column.embedding_column(
        dimension=10,
        categorical_column=tf.feature_column.categorical_column_with_vocabulary_list(
        key="GarageCond",
        vocabulary_list=train_df.GarageCond.unique().tolist()
    )),
    tf.feature_column.embedding_column(
        dimension=10,
        categorical_column=tf.feature_column.categorical_column_with_vocabulary_list(
        key="PavedDrive",
        vocabulary_list=train_df.PavedDrive.unique().tolist()
    )),
    tf.feature_column.numeric_column(
        key="WoodDeckSF",
        dtype=tf.int64
    ),
    tf.feature_column.numeric_column(
        key="OpenPorchSF",
        dtype=tf.int64
    ),
    tf.feature_column.numeric_column(
        key="EnclosedPorch",
        dtype=tf.int64
    ),
    tf.feature_column.numeric_column(
        key="3SsnPorch",
        dtype=tf.int64
    ),
    tf.feature_column.numeric_column(
        key="ScreenPorch",
        dtype=tf.int64
    ),
    tf.feature_column.numeric_column(
        key="PoolArea",
        dtype=tf.int64
    ),
    tf.feature_column.embedding_column(
        dimension=10,
        categorical_column=tf.feature_column.categorical_column_with_vocabulary_list(
        key="PoolQC",
        vocabulary_list=train_df.PoolQC.unique().tolist()
    )),
    tf.feature_column.embedding_column(
        dimension=10,
        categorical_column=tf.feature_column.categorical_column_with_vocabulary_list(
        key="Fence",
        vocabulary_list=train_df.Fence.unique().tolist()
    )),
    tf.feature_column.embedding_column(
        dimension=10,
        categorical_column=tf.feature_column.categorical_column_with_vocabulary_list(
        key="MiscFeature",
        vocabulary_list=train_df.MiscFeature.unique().tolist()
    )),
    tf.feature_column.numeric_column(
        key="MiscVal",
        dtype=tf.int64
    ),
    tf.feature_column.numeric_column(
        key="MoSold",
        dtype=tf.int64
    ),
    tf.feature_column.numeric_column(
        key="YrSold",
        dtype=tf.int64
    ),
    tf.feature_column.embedding_column(
        dimension=10,
        categorical_column=tf.feature_column.categorical_column_with_vocabulary_list(
        key="SaleType",
        vocabulary_list=train_df.SaleType.unique().tolist()
    )),
    tf.feature_column.embedding_column(
        dimension=10,
        categorical_column=tf.feature_column.categorical_column_with_vocabulary_list(
            key="SaleCondition",
            vocabulary_list=train_df.SaleCondition.unique().tolist()
        ),
    ),
]
from tensorflow.python.ops import nn
classifier = tf.estimator.DNNRegressor(
    feature_columns=tf_features,
    hidden_units=[128, 128, 64, 64,],
    # dropout=0.1,
    activation_fn=nn.relu,
    optimizer=tf.train.AdamOptimizer(),
    # batch_norm=True,
    model_dir="./tf_dnn_regressor",
    # loss_reduction=tf.losses.Reduction.SUM
)
print(classifier.train(input_fn=create_train_input_fn))
print(classifier.evaluate(input_fn=create_train_input_fn))


print("\n-------------------------------\n")
# print(test_df.to_dict(orient="records"))

results = classifier.predict(input_fn=create_test_input_fn)
predictions = []
for result in results:
    # print(result)
    if math.isnan(result["predictions"][0]):
        print(result)
    predictions.append(result["predictions"][0])
# predictions = [
#     result["predictions"][0] for result in results
# ]

test_df = test_df.assign(
    SalePrice=predictions
)
print(test_df[pd.isnull(test_df.SalePrice)])
print(test_df[["Id","SalePrice"]])
test_df[["Id", "SalePrice"]].fillna(test_df.SalePrice.mean()).to_csv(
    "test_submission.csv",
    index=False
)
