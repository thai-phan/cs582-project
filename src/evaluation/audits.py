import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from src.evaluation.metrics import metrics
from src.evaluation.figures import save


def leakage_control(dataset,split,honest_test_row,out):
    frame=dataset.frame; y=dataset.labels()
    model=DecisionTreeClassifier(max_depth=1,random_state=42)
    model.fit(frame.loc[split.train,['close_value']],y.loc[split.train])
    p=model.predict_proba(frame.loc[split.test,['close_value']])[:,1]
    leaky={'model':'LEAKY_close_value_stump',**metrics(y.loc[split.test],p),'purpose':'Invalid future information; illustration only'}
    honest=dict(honest_test_row,purpose='Pre-close feature set; no outcome predictors')
    pd.DataFrame([honest,leaky]).to_csv(out.checks/'leakage_audit.csv',index=False)
    fig,ax=plt.subplots(figsize=(7,4))
    ax.bar(['Honest selected model','LEAKY close_value'],[honest['roc_auc'],leaky['roc_auc']],color=['#278b9c','#db6849'])
    ax.set(ylim=(0,1.08),ylabel='Test ROC-AUC',title='Why outcome fields must be excluded')
    save(fig,out.figures/'leakage_audit.png')


def priority_checks(y,p,yval,pval,out):
    def groups(y,p,low,high):
        frame=pd.DataFrame({'observed':np.asarray(y),'probability':p})
        frame['priority']=np.where(p>=high,'High',np.where(p>=low,'Medium','Low'))
        result=frame.groupby('priority').agg(count=('observed','size'),observed_win_rate=('observed','mean'),mean_probability=('probability','mean')).reindex(['Low','Medium','High'])
        result['count']=result['count'].fillna(0).astype(int)
        return result.reset_index()
    groups(y,p,.4,.7).to_csv(out.checks/'priority_test.csv',index=False)
    sensitivity=[]
    for low,high in [(.3,.6),(.4,.7),(.5,.8)]:
        table=groups(yval,pval,low,high); table['medium_min']=low; table['high_min']=high
        sensitivity.append(table)
    pd.concat(sensitivity).to_csv(out.checks/'priority_validation_sensitivity.csv',index=False)
