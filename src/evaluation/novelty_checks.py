import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score,brier_score_loss,roc_auc_score
from sklearn.model_selection import train_test_split
from src.models.zoo import MODEL_NAMES,fit_model
from src.explain.reference import reference_sensitivities

LEARNED=[name for name in MODEL_NAMES if name!='dummy_prior']
TOP_K=3


def bootstrap_auc(y,p,draws,seed=42):
    y=np.asarray(y); p=np.asarray(p); rng=np.random.default_rng(seed); values=[]
    for _ in range(draws):
        i=rng.integers(0,len(y),len(y))
        if len(np.unique(y[i]))==2: values.append(roc_auc_score(y[i],p[i]))
    return float(np.percentile(values,2.5)),float(np.percentile(values,97.5))


def paired_auc_delta(y,p_new,p_base,draws,seed=42):
    y=np.asarray(y); rng=np.random.default_rng(seed); values=[]
    for _ in range(draws):
        i=rng.integers(0,len(y),len(y))
        if len(np.unique(y[i]))==2: values.append(roc_auc_score(y[i],p_new[i])-roc_auc_score(y[i],p_base[i]))
    return float(np.percentile(values,2.5)),float(np.percentile(values,97.5))


def protocol_indices(dataset,split,seeds):
    frame=dataset.frame; engage=frame.engage_date; y=dataset.labels()
    yield 'asof_purged',42,split.train,split.validation,split.test
    train=frame.index[engage<split.validation_start].to_numpy()
    validation=frame.index[(engage>=split.validation_start)&(engage<split.test_start)].to_numpy()
    yield 'chronological_no_purge',42,train,validation,split.test
    for seed in seeds:
        train,rest=train_test_split(frame.index.to_numpy(),test_size=.4,stratify=y,random_state=seed)
        validation,test=train_test_split(rest,test_size=.5,stratify=y.loc[rest],random_state=seed)
        yield 'random_stratified',seed,train,validation,test


def split_protocols(dataset,split,out,quick=False):
    X=dataset.features(); y=dataset.labels(); draws=50 if quick else 1000
    seeds=range(1) if quick else range(5); rows=[]; asof={}
    for protocol,seed,train,validation,test in protocol_indices(dataset,split,seeds):
        yt=y.loc[test].to_numpy()
        for name in LEARNED:
            p=fit_model(name,dataset,train,validation,quick=quick).predict_proba(X.loc[test])[:,1]
            low,high=bootstrap_auc(yt,p,draws)
            row=dict(protocol=protocol,seed=seed,model=name,train_rows=len(train),validation_rows=len(validation),
                     test_rows=len(test),test_win_rate=float(yt.mean()),roc_auc=float(roc_auc_score(yt,p)),
                     auc_ci_low=low,auc_ci_high=high,average_precision=float(average_precision_score(yt,p)),
                     brier=float(brier_score_loss(yt,p)),paired_delta_ci_low=np.nan,paired_delta_ci_high=np.nan)
            if protocol=='asof_purged': asof[name]=p
            if protocol=='chronological_no_purge':
                row['paired_delta_ci_low'],row['paired_delta_ci_high']=paired_auc_delta(yt,p,asof[name],draws)
            rows.append(row)
    table=pd.DataFrame(rows); table.to_csv(out.checks/'split_protocol_comparison.csv',index=False)
    summary=table.groupby(['protocol','model'],sort=False).agg(runs=('roc_auc','size'),train_rows=('train_rows','mean'),
        test_rows=('test_rows','mean'),roc_auc_mean=('roc_auc','mean'),roc_auc_sd=('roc_auc','std'),
        auc_ci_low_mean=('auc_ci_low','mean'),auc_ci_high_mean=('auc_ci_high','mean')).reset_index()
    reference=summary.loc[summary.protocol.eq('asof_purged')].set_index('model').roc_auc_mean
    summary['delta_vs_asof']=summary.roc_auc_mean-summary.model.map(reference)
    summary.to_csv(out.checks/'split_protocol_summary.csv',index=False)
    return summary


def feature_groups(fitted,dataset):
    encoder=fitted.prep.named_transformers_['cat'].named_steps['encode']; groups={}; start=0
    for col,categories in zip(dataset.categorical,encoder.categories_):
        groups[col]=list(range(start,start+len(categories))); start+=len(categories)
    for col in dataset.numeric:
        groups[col]=[start]; start+=1
    if start!=len(fitted.prep.get_feature_names_out()): raise AssertionError('unexpected preprocessor layout')
    return groups


def rf_agreement(rf,dataset,X,reference,out):
    import shap
    transformed=np.asarray(rf.prep.transform(X),dtype=np.float32)
    explainer=shap.TreeExplainer(rf.clf,feature_perturbation='tree_path_dependent',model_output='raw')
    values=np.asarray(explainer.shap_values(transformed,check_additivity=True))[:,:,1]
    groups=feature_groups(rf,dataset)
    grouped=pd.DataFrame({col:values[:,idx].sum(axis=1) for col,idx in groups.items()},index=X.index)[list(X.columns)]
    sensitivity=reference_sensitivities(rf,X,reference)[list(X.columns)]
    rows=[]
    for idx in X.index:
        s=grouped.loc[idx]; r=sensitivity.loc[idx]
        top_s=set(s.abs().nlargest(TOP_K).index); top_r=set(r.abs().nlargest(TOP_K).index)
        lead=s.abs().idxmax()
        rows.append(dict(spearman=float(spearmanr(s,r).statistic),top_overlap=len(top_s&top_r)/TOP_K,
                         top_shap_feature=lead,top_feature_same_sign=bool(np.sign(s[lead])==np.sign(r[lead]))))
    table=pd.DataFrame(rows); table.insert(0,'row',X.index); table.to_csv(out.checks/'explanation_agreement_rf.csv',index=False)
    n=len(X.columns)
    return dict(model='random_forest_raw',rows=len(X),features=n,top_k=TOP_K,
                spearman_median=float(table.spearman.median()),spearman_mean=float(table.spearman.mean()),
                top_overlap_mean=float(table.top_overlap.mean()),top_overlap_random_expectation=TOP_K/n,
                top_feature_same_sign_rate=float(table.top_feature_same_sign.mean()))


def deletion_test(model,X,reference,quick=False,seed=42):
    p=model.predict_proba(X)[:,1]; sensitivity=reference_sensitivities(model,X,reference)[list(X.columns)]
    rng=np.random.default_rng(seed); n,m=len(X),X.shape[1]
    def replace(mask):
        changed=X.copy()
        for j,col in enumerate(X.columns):
            changed[col]=np.where(mask[:,j],reference[col],changed[col].to_numpy(dtype=object))
        return np.abs(p-model.predict_proba(changed)[:,1])
    order=np.argsort(-sensitivity.abs().to_numpy(),axis=1,kind='stable')[:,:TOP_K]
    mask=np.zeros((n,m),dtype=bool); mask[np.arange(n)[:,None],order]=True
    top=replace(mask); draws=[]
    for _ in range(3 if quick else 20):
        chosen=np.argsort(rng.random((n,m)),axis=1)[:,:TOP_K]
        mask=np.zeros((n,m),dtype=bool); mask[np.arange(n)[:,None],chosen]=True
        draws.append(replace(mask))
    random=np.mean(draws,axis=0)
    return dict(model=getattr(model,'name','model'),rows=n,top_k=TOP_K,random_draws=len(draws),
                mean_abs_change_top=float(top.mean()),mean_abs_change_random=float(random.mean()),
                ratio_top_to_random=float(top.mean()/random.mean()) if random.mean()>0 else None,
                share_rows_top_exceeds_random=float((top>random).mean()))


def explanation_checks(rf,selected,dataset,split,reference,out,quick=False):
    X=dataset.features(); xt=X.loc[split.test]
    result=dict(agreement=rf_agreement(rf,dataset,xt.iloc[:(12 if quick else 64)],reference,out),
                deletion=deletion_test(selected,xt,reference,quick=quick),
                note='Checks whether explanations describe model behaviour; they do not test causal effects.')
    (out.checks/'explanation_checks.json').write_text(json.dumps(result,indent=2))
    return result
