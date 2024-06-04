import glob
import os
from datetime import datetime

import dill
import pandas as pd
import json

path = os.environ.get('PROJECT_PATH', '.')
def predict():

    # Определяем последнюю модель
    latest_model = sorted(os.listdir(f'{path}/data/models'))[-1]
    # Загружаем обученную модель
    with open(f'{path}/data/models/{latest_model}', 'rb') as file:
        model = dill.load(file)

    preds = pd.DataFrame(columns=['car_id', 'pred'])

    for file in glob.glob(f'{path}/data/test/*.json'):
        with open(file) as fin:
            form = json.load(fin)
            df = pd.DataFrame.from_dict([form])
            y = model.predict(df)
            x = {'car_id': df.id, 'pred': y}
            df1 = pd.DataFrame(x)
            preds = pd.concat([preds, df1], axis = 0)
    print(preds)

    preds.to_csv(f'{path}/data/predictions/preds_{datetime.now().strftime("%Y%m%d%H%M")}.csv', index = False)



if __name__ == '__main__':
    predict()
