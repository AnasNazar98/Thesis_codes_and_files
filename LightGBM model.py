


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import itertools as itr
from skimpy import skim
from scipy.stats import iqr
from sklearn.model_selection import train_test_split
from feature_engine.timeseries.forecasting import LagFeatures
from feature_engine.timeseries.forecasting import WindowFeatures
from feature_engine.timeseries.forecasting import ExpandingWindowFeatures
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import median_absolute_error

import tensorflow as tf
import random



import os
import time
day_number = 7


SEED = 99
tf.random.set_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

garmin = pd.read_excel('C:/Users/anasn/Desktop/E/Semester 4/Thesis/files/Garmin_days_EMA_Anas.xlsx',
                       index_col=0)
ema = pd.read_csv('C:/Users/anasn/Desktop/E/Semester 4/Thesis/files/EMA_days_Answered_Final.csv'
                  , sep=';'
                  , decimal=',')

garmin_valid_ids = garmin[garmin['day'] == 14]['participant_id'].unique()

garmin = (garmin
          .query("day <= 14 and participant_id in @garmin_valid_ids"))


garmin = (garmin
          .groupby(['participant_id', 'day', 'date', 'hours_cat'])
          .agg(Steps = ("Steps", lambda x: np.sum(x)))
          .sort_values(['participant_id', 'date', 'hours_cat'])
          .reset_index(drop=False))

garmin['hours_cat'] = pd.Categorical(garmin['hours_cat']
     , categories=['Morning', 'Noon', 'Afternoon', 'Evening'])

garmin = (garmin
          .sort_values(['participant_id', 'day', 'date', 'hours_cat']))



participant_id = garmin['participant_id'].unique()
day = np.arange(1, 15)
hours_cat = garmin['hours_cat'].unique()

template = pd.DataFrame(list(itr.product(participant_id, day, hours_cat)), 
                        columns=['participant_id', 'day', 'hours_cat'])

template['timestep'] = (template
                        .groupby('participant_id')
                        .cumcount() + 1)

template = pd.merge(template, garmin, on=["participant_id", "day", "hours_cat"]
                    , how='left')

garmin = template.copy()

ema["Time_cat"] = pd.Categorical(ema['Time_cat'],
    categories=['Morning', 'Noon', 'Afternoon', 'Evening'])

ema = (ema
       .rename(columns = {"Time_cat": "hours_cat"}))

garmin = pd.merge(garmin, ema, how='left',
                  on=["participant_id", "day", "hours_cat"])


garmin['date'] = (garmin
                  .groupby(["participant_id", "day"])['date']
                  .transform(lambda x: x.ffill().bfill()))

garmin.columns


garmin = (garmin
          .get(['participant_id', 'day', 'hours_cat', 'timestep', 'date',
                 'PHYSICAL_NORM', 'MENTAL_NORM', 'MOTIVATION_NORM', 'EFFICACY_NORM',
                 'CONTEXT_NORM', 'Steps']))







np.random.seed(SEED)
shuffled_ids = np.random.permutation(participant_id)  
n = len(shuffled_ids)

train_size = int(np.floor(0.7 * n))
val_size = int(np.floor(0.1 * n))

train_ids = shuffled_ids[:train_size]
val_ids = shuffled_ids[train_size:train_size + val_size]
test_ids = shuffled_ids[train_size + val_size:]

print(len(train_ids), len(val_ids), len(test_ids))
print(sorted(train_ids))
print(sorted(val_ids))
print(sorted(test_ids))




###############################################################################
# Yeo-Johnson

from feature_engine.transformation import YeoJohnsonTransformer


steps_train_df = garmin[garmin['participant_id'].isin(train_ids)][['Steps']].dropna()
step_transformer = YeoJohnsonTransformer(variables=['Steps'])
step_transformer.fit(steps_train_df)

garmin['Steps_original'] = garmin['Steps']

steps_non_null = garmin.loc[garmin['Steps'].notna(), ['Steps']]
transformed_steps = step_transformer.transform(steps_non_null)

garmin['Steps_transformed'] = np.nan
garmin.loc[steps_non_null.index, 'Steps_transformed'] = transformed_steps['Steps']

garmin['Steps'] = garmin['Steps_transformed']


###############################################################################
lag_vars = ['Steps'
                #, "PHYSICAL_NORM", "MENTAL_NORM", "MOTIVATION_NORM", "EFFICACY_NORM", "CONTEXT_NORM"
]

length = 4*day_number
lag_range = np.arange(1, length+1).tolist()


hours_map = {'Morning': 0, 'Noon': 1, 'Afternoon': 2, 'Evening': 3}
garmin['hours_idx'] = garmin['hours_cat'].map(hours_map)

garmin = pd.concat([garmin, pd.get_dummies(garmin['hours_cat'])], axis=1)
garmin[['Morning', 'Noon', 'Afternoon', 'Evening']] = garmin[['Morning', 'Noon', 'Afternoon', 'Evening']].astype(int)


def make_lag(df):
    lf = LagFeatures(periods=lag_range 
                     #list(range(1, length+1))
                     , variables=lag_vars
                     , missing_values='ignore')
    return lf.fit_transform(df)





garmin = (
    garmin
    .groupby(['participant_id'])
    .apply(make_lag)
    .reset_index(drop=True)
    )

garmin.columns


###############################################################################
# multi step 
for i in range(0, 4):
    garmin[f'Steps_t{i}'] = garmin.groupby('participant_id')['Steps'].shift(-i)
    garmin[f'Steps_original_t{i}'] = garmin.groupby('participant_id')['Steps_original'].shift(-i)



target_cols = [f'Steps_t{i}' for i in range(0, 4)]

target_original_cols = [f'Steps_original_t{i}' for i in range(4)]

no_missing = garmin[target_original_cols].notna().all(axis=1)

data_train = garmin[
    garmin['participant_id'].isin(train_ids) &
    (garmin['timestep'] > length) &
    no_missing
]

data_val = garmin[
    garmin['participant_id'].isin(val_ids) &
    (garmin['timestep'] > length) &
    no_missing
]

data_test = garmin[
    garmin['participant_id'].isin(test_ids) &
    (garmin['timestep'] > length) &
    no_missing
]






lagged_features = garmin.filter(regex=r"_lag_\d+$").columns.tolist()






other_features = ['hours_cat']
time_of_day_features = ['Noon', 'Afternoon', 'Evening']


features = (time_of_day_features
            + lagged_features)


X_train = (data_train
           .get(features + ['participant_id']))
y_train = data_train.loc[:, target_cols]

X_val = (data_val
           .get(features + ['participant_id']))
y_val = data_val.loc[:, target_cols]


X_test = (data_test
          .get(features + ['participant_id']))
y_test = data_test.loc[:, target_cols]


###############################################################################

from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error
import lightgbm as lgb



base_model = lgb.LGBMRegressor(
    n_estimators=3000,
    num_leaves=1000,
    max_depth=100,
    min_child_samples=1,  
    min_split_gain=0,
    #subsample=1,
    learning_rate=0.005,
    reg_alpha=0.01,
    reg_lambda=0.01,

    objective='regression_l1',

    random_state=123,
    n_jobs=-1,
    verbosity=-1   
)



model = MultiOutputRegressor(base_model)

model.fit(X_train, y_train
          )

y_pred_train = model.predict(X_train)
y_pred_val = model.predict(X_val)
y_pred_test = model.predict(X_test)

from sktime.performance_metrics.forecasting import MedianAbsolutePercentageError
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score

def evaluate(y_true, y_pred, name=""):
    #y_true = pd.Series(y_true).reset_index(drop=True)
    #y_pred = pd.Series(y_pred).reset_index(drop=True)

    mae = mean_absolute_error(y_true, y_pred)
    medae = median_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mean_val = np.mean(y_true)
    median_val = np.median(y_true)

    

    print(f"\n{name} Set Evaluation:")
    print(f"MAE:            {mae:.2f}")
    print(f"MedAE:          {medae:.2f}")
    print(f"R2:             {r2:.2f}")
    print(f"Mean:           {mean_val:.2f}")
    print(f"Median:         {median_val:.2f}")
    print(f"MAE / Mean:     {mae / mean_val:.3f}")
    print(f"MedAE / Median: {medae / median_val:.3f}")
    
    
    return {
            'MAE': round(mae, 2),
            'MedAE': round(medae, 2),
            'R2': round(r2, 2),
            'Mean': round(mean_val, 2),
            'Median': round(median_val, 2),
            'MAE/Mean': round(mae / mean_val, 3),
            'MedAE/Median': round(medae / median_val, 3)
        }

y_train_inv = pd.DataFrame()
y_pred_train_inv = pd.DataFrame()
y_val_inv = pd.DataFrame()
y_pred_val_inv = pd.DataFrame()
y_test_inv = pd.DataFrame()
y_pred_test_inv = pd.DataFrame()

for i, col in enumerate(y_train.columns):
    col_train = pd.DataFrame({'Steps': y_train.iloc[:, i]})
    col_pred_train = pd.DataFrame({'Steps': y_pred_train[:, i]})
    
    col_val = pd.DataFrame({'Steps': y_val.iloc[:, i]})
    col_pred_val = pd.DataFrame({'Steps': y_pred_val[:, i]})
    
    col_test = pd.DataFrame({'Steps': y_test.iloc[:, i]})
    col_pred_test = pd.DataFrame({'Steps': y_pred_test[:, i]})

    y_train_inv[f'Steps_t{i+1}'] = step_transformer.inverse_transform(col_train)['Steps']
    y_pred_train_inv[f'Steps_t{i+1}'] = step_transformer.inverse_transform(col_pred_train)['Steps']

    y_val_inv[f'Steps_t{i+1}'] = step_transformer.inverse_transform(col_val)['Steps']
    y_pred_val_inv[f'Steps_t{i+1}'] = step_transformer.inverse_transform(col_pred_val)['Steps']

    y_test_inv[f'Steps_t{i+1}'] = step_transformer.inverse_transform(col_test)['Steps']
    y_pred_test_inv[f'Steps_t{i+1}'] = step_transformer.inverse_transform(col_pred_test)['Steps']




# Run evaluations
#evaluate(y_train_inv, y_pred_train_inv, name="Train")
#evaluate(y_val_inv, y_pred_val_inv, name="Validation")
evaluate(y_test_inv, y_pred_test_inv, name="Test")

metrics = evaluate(y_test_inv, y_pred_test_inv, name="Test")
###############################################################################
###############################################################################
###############################################################################
###############################################################################


# ploting 
import seaborn as sns


save_path = "C:/Users/anasn/Desktop/E/Semester 4/Thesis/R code/xgboost timeseries/prediction plots xgboost/results/4step/"
os.makedirs(save_path, exist_ok=True)

steps = [f'Steps_t{i+1}' for i in range(4)]

test_mask = data_test['participant_id'].isin(test_ids)
multi_target_test = data_test[test_mask].reset_index(drop=True)

y_test_inv_with_id = y_test_inv.copy()
y_test_inv_with_id['participant_id'] = data_test['participant_id'].values

y_pred_test_inv_with_id = y_pred_test_inv.copy()
y_pred_test_inv_with_id['participant_id'] = data_test['participant_id'].values

for idx, pid in enumerate(y_test_inv_with_id['participant_id'].unique(), 1):
    print(f" participant {pid} ({idx}/{y_test_inv_with_id['participant_id'].nunique()})")

    true_vals = y_test_inv_with_id[y_test_inv_with_id['participant_id'] == pid][steps].reset_index(drop=True)
    pred_vals = y_pred_test_inv_with_id[y_pred_test_inv_with_id['participant_id'] == pid][steps].reset_index(drop=True)

    for i in range(len(true_vals)):
        df_plot = pd.DataFrame({
            'Time Step': steps * 2,
            'Steps': list(true_vals.loc[i]) + list(pred_vals.loc[i]),
            'Type': ['Actual'] * 4 + ['Predicted'] * 4
        })

        plt.figure(figsize=(8, 5))
        sns.lineplot(data=df_plot, x='Time Step', y='Steps', hue='Type', marker='o')

        plt.title(f'Participant {pid} - Prediction {i+1}')
        plt.xlabel('Time Step')
        plt.ylabel('Steps')
        plt.grid(True)
        plt.legend()

        filename = f"participant_{pid}_prediction_{i+1}.png"
        plt.savefig(os.path.join(save_path, filename)
                    , dpi=100)
        plt.close()