from dataclasses import dataclass
import numpy as np
import pandas as pd
from src.data.dataset import Dataset


@dataclass
class Split:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    validation_start: pd.Timestamp
    test_start: pd.Timestamp
    manifest: pd.DataFrame


def asof_split(dataset: Dataset) -> Split:
    frame = dataset.frame
    if frame[['engage_date','close_date']].isna().any().any():
        raise ValueError('as-of split requires complete engagement and outcome dates')
    dates = frame['engage_date'].sort_values(kind='stable')
    if len(dates) < 10:
        raise ValueError('not enough dated observations')
    validation_start = dates.iloc[int(len(dates) * .6)]
    test_start = dates.iloc[int(len(dates) * .8)]
    if validation_start >= test_start:
        raise ValueError('insufficient distinct dates for chronological periods')
    engage, close = frame['engage_date'], frame['close_date']
    train = (engage < validation_start) & (close < validation_start)
    validation = (engage >= validation_start) & (engage < test_start) & (close < test_start)
    test = engage >= test_start
    role = pd.Series('purged_train', index=frame.index)
    role.loc[(engage >= validation_start) & (engage < test_start)] = 'purged_validation'
    role.loc[train],role.loc[validation],role.loc[test] = 'train','validation','test'
    manifest = frame[['opportunity_id','engage_date','close_date']].copy()
    manifest['role'] = role
    indices = [frame.index[mask].to_numpy() for mask in (train,validation,test)]
    for name,idx in zip(['train','validation','test'],indices):
        if len(idx) == 0 or dataset.labels().loc[idx].nunique() != 2:
            raise ValueError(f'{name} needs observations from both classes')
    return Split(*indices,validation_start,test_start,manifest)
