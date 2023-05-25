from calculate_gradients import DB, KEY_PREFIX
from tqdm import tqdm
import pandas as pd

correct_value = 6.0

database_keys = DB.keys(KEY_PREFIX + '*')
database_keys.sort(key=lambda x: int(x.split('_')[-1]))

df = pd.DataFrame()
for key in tqdm(database_keys):
    info = DB.hgetall(key)

    for k, v in info.items():
        # Set the row to the key
        df.loc[key, k] = v

    df.loc[key, 'diff'] = float(info['y_hat']) - correct_value


df.to_csv(f'{KEY_PREFIX}_results.csv')