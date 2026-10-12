import json
import numpy as np
import pandas as pd
from src.explain.priority import priority_group


def make_reference(X,dataset):
    reference={}
    for col in dataset.categorical:
        mode=X[col].dropna().mode()
        reference[col]=mode.iloc[0] if len(mode) else 'Missing'
    for col in dataset.numeric:
        value=X[col].median()
        reference[col]=float(value) if pd.notna(value) else 0.
    return reference


def reference_sensitivities(model,X,reference):
    p=model.predict_proba(X)[:,1]
    effect=pd.DataFrame(index=X.index)
    for col in X:
        replaced=X.copy(); replaced[col]=reference[col]
        effect[col]=p-model.predict_proba(replaced)[:,1]
    return effect


CONTEXT=['opportunity_id','deal_stage','engage_date','account','product','sales_agent']


def explain_rows(model,rows,features,reference,threshold,scoring_context='frozen_model_snapshot_demo'):
    if not 0<=threshold<=1: raise ValueError('invalid decision threshold')
    missing=[col for col in features if col not in rows]
    if missing: raise ValueError(f'input lacks feature columns: {missing}')
    X=rows[features]
    p=model.predict_proba(X)[:,1]
    effect=reference_sensitivities(model,X,reference)
    result=rows[[col for col in CONTEXT if col in rows]].copy()
    result['win_probability']=p; result['loss_probability']=1-p
    result['predicted_outcome']=np.where(p>=threshold,'Won','Lost')
    result['priority']=[priority_group(float(v)) for v in p]
    result['account_missing']=(rows['account'] if 'account' in rows else rows['sector']).isna().to_numpy()
    result['input_warning']=np.where(result.account_missing,'Account missing: numeric imputation / unknown categories','')
    result['model']=model.name
    result['explanation_method']='single_feature_reference_sensitivity'
    result['scoring_context']=scoring_context
    result['decision_threshold']=threshold
    result['priority_medium_min']=.40; result['priority_high_min']=.70
    positives=[]; negatives=[]; raw=[]
    for idx,values in effect.iterrows():
        def describe(selected):
            return '; '.join(f'{col}={X.loc[idx,col]} (delta P(Won)={value:+.4f})' for col,value in selected.items())
        positives.append(describe(values[values>1e-10].nlargest(3)) or 'No positive reference sensitivity')
        negatives.append(describe(values[values < -1e-10].nsmallest(3)) or 'No negative reference sensitivity')
        raw.append(json.dumps({k:float(v) for k,v in values.items()},sort_keys=True))
    result['positive_factors']=positives; result['negative_factors']=negatives
    result['reference_deltas_json']=raw
    return result.sort_values('win_probability',ascending=False,kind='stable').reset_index(drop=True)


def explain_open(model,dataset,reference,threshold):
    return explain_rows(model,dataset.extra['scorable_open_deals'],dataset.feature_columns,reference,threshold)
