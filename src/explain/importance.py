import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from src.evaluation.figures import save


def native_importances(models,out):
    for name in ['logistic_regression','random_forest','tabnet']:
        model=models[name]
        if name=='logistic_regression':
            values=model.clf.coef_[0]
            description='Signed log-odds coefficient; numerics standardized'
        else:
            values=model.clf.feature_importances_
            description='Impurity importance' if name=='random_forest' else 'TabNet attention importance'
        table=pd.DataFrame({'feature':model.prep.get_feature_names_out(),'importance':values})
        table['absolute_importance']=table.importance.abs()
        table=table.sort_values('absolute_importance',ascending=False)
        table.to_csv(out.explain/f'feature_importance_{name}.csv',index=False)
        shown=table.head(12).iloc[::-1]
        fig,ax=plt.subplots(figsize=(9,5))
        ax.barh(shown.feature,shown.importance,color='#278b9c')
        ax.set(title=name,xlabel=description); save(fig,out.figures/f'importance_{name}.png')


def permutation_importance(model,X,y,out,quick=False):
    rng=np.random.default_rng(42); base=roc_auc_score(y,model.predict_proba(X)[:,1]); rows=[]
    for col in X:
        drops=[]
        for _ in range(1 if quick else 3):
            changed=X.copy(); changed[col]=rng.permutation(changed[col].to_numpy())
            drops.append(base-roc_auc_score(y,model.predict_proba(changed)[:,1]))
        rows.append({'feature':col,'mean_auc_drop':float(np.mean(drops)),'std_auc_drop':float(np.std(drops))})
    result=pd.DataFrame(rows).sort_values('mean_auc_drop',ascending=False)
    result.to_csv(out.explain/'permutation_importance_test.csv',index=False)
    fig,ax=plt.subplots(figsize=(8,5)); shown=result.head(12).iloc[::-1]
    ax.barh(shown.feature,shown.mean_auc_drop,xerr=shown.std_auc_drop,color='#278b9c')
    ax.set(title='Held-out permutation sensitivity',xlabel='ROC-AUC drop; negative values are possible')
    save(fig,out.figures/'permutation_importance.png')


def shap_audit(rf,X,ids,out,quick=False):
    import shap
    sample=X.iloc[:(12 if quick else 64)]
    transformed=np.asarray(rf.prep.transform(sample),dtype=np.float32)
    explainer=shap.TreeExplainer(rf.clf,feature_perturbation='tree_path_dependent',model_output='raw')
    values=explainer.shap_values(transformed,check_additivity=True)
    values=np.asarray(values)[:,:,1]
    expected=float(np.asarray(explainer.expected_value)[1])
    prediction=rf.clf.predict_proba(transformed)[:,1]
    error=float(np.max(np.abs(expected+values.sum(axis=1)-prediction)))
    if error>1e-5: raise AssertionError('RF SHAP additivity check failed')
    names=rf.prep.get_feature_names_out()
    global_table=pd.DataFrame({'feature':names,'mean_absolute_shap':np.abs(values).mean(axis=0)}).sort_values('mean_absolute_shap',ascending=False)
    global_table.to_csv(out.explain/'rf_shap_global.csv',index=False)
    local=pd.DataFrame(values,columns=names); local.insert(0,'opportunity_id',np.asarray(ids)[:len(sample)])
    local.to_csv(out.explain/'rf_shap_values.csv',index=False)
    audit={'model':'random_forest_raw','sample':'first dated test rows','rows':len(sample),
           'expected_probability':expected,'max_additivity_error':error,'method':'TreeSHAP tree_path_dependent',
           'warning':'Associations, not causal effects; not the calibrated scoring model'}
    (out.explain/'shap_audit.json').write_text(json.dumps(audit,indent=2))
    fig,ax=plt.subplots(figsize=(9,5)); shown=global_table.head(12).iloc[::-1]
    ax.barh(shown.feature,shown.mean_absolute_shap,color='#278b9c')
    ax.set(title='Random Forest: TreeSHAP on test sample',xlabel='Mean absolute SHAP contribution to P(Won)')
    save(fig,out.figures/'rf_shap.png')
    local0=pd.Series(values[0],index=names); keep=local0.abs().nlargest(10).index
    shown=local0.loc[keep].sort_values()
    fig,ax=plt.subplots(figsize=(9,5)); ax.barh(shown.index,shown,color=np.where(shown>=0,'#278b9c','#db6849'))
    ax.set(title=f'RF explanation: {np.asarray(ids)[0]} (test example)',xlabel='SHAP contribution; association, not cause')
    save(fig,out.figures/'rf_shap_local.png')
