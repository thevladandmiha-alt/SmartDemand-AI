"""Reproducible educational demand forecasting experiment."""
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

FEATURES = ['День недели', 'Цена', 'Скидка (%)', 'Реклама']
TARGET = 'Продажи'

def demo_data(n=420, seed=42):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    weekday = t % 7
    price = rng.uniform(80, 120, n)
    discount = rng.choice([0, 5, 10, 15, 20], n)
    ads = rng.integers(0, 2, n)
    sales = np.maximum(0, 110 - .55*price + 1.3*discount + 12*ads + 9*np.isin(weekday,[5,6]) + 7*np.sin(t/14) + rng.normal(0,5,n))
    return pd.DataFrame({'День':t, 'День недели':weekday, 'Цена':price, 'Скидка (%)':discount, 'Реклама':ads, 'Продажи':np.round(sales)})

def validate_data(df):
    missing = [c for c in FEATURES+[TARGET] if c not in df.columns]
    if missing:
        raise ValueError('Отсутствуют столбцы: '+', '.join(missing))
    out = df.copy()
    for col in FEATURES+[TARGET]:
        out[col] = pd.to_numeric(out[col], errors='coerce')
    if out[FEATURES+[TARGET]].isna().any().any():
        raise ValueError('В обязательных столбцах есть пустые или нечисловые значения')
    if len(out) < 70:
        raise ValueError('Необходимо минимум 70 строк, упорядоченных по времени')
    if not out['День недели'].between(0,6).all() or not out['Реклама'].isin([0,1]).all():
        raise ValueError('День недели: 0–6; реклама: 0 или 1')
    if (out[['Цена','Скидка (%)','Продажи']] < 0).any().any() or (out['Скидка (%)'] > 100).any():
        raise ValueError('Цена и продажи не могут быть отрицательными; скидка от 0 до 100')
    return out.reset_index(drop=True)

def experiment(df):
    df = validate_data(df)
    split = int(len(df)*.8)
    train, test = df.iloc[:split], df.iloc[split:]
    model = make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(24,12), activation='relu', solver='adam', max_iter=2500, early_stopping=True, random_state=42))
    model.fit(train[FEATURES], train[TARGET])
    pred = np.maximum(0, model.predict(test[FEATURES]))
    baseline = np.repeat(train[TARGET].mean(), len(test))
    result = pd.DataFrame({'Факт':test[TARGET].to_numpy(), 'Нейросеть':pred, 'Среднее обучения':baseline})
    if 'День' in df.columns:
        result.insert(0,'День',test['День'].to_numpy())
    metrics = {
        'mae_model':float(mean_absolute_error(result['Факт'],pred)),
        'mae_baseline':float(mean_absolute_error(result['Факт'],baseline)),
        'rmse_model':float(np.sqrt(mean_squared_error(result['Факт'],pred))),
        'test_count':len(test), 'train_count':len(train),
    }
    metrics['improvement_pct'] = (1-metrics['mae_model']/metrics['mae_baseline'])*100 if metrics['mae_baseline'] else float('nan')
    return model, result, metrics

if __name__ == '__main__':
    model, result, m = experiment(demo_data())
    for k,v in m.items(): print(f'{k}: {v:.4f}' if isinstance(v,float) else f'{k}: {v}')
