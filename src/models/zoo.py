from copy import deepcopy
from dataclasses import dataclass
from time import perf_counter
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from threadpoolctl import threadpool_limits
from src.data.preprocess import make_preprocessor

MODEL_NAMES=('dummy_prior','logistic_regression','random_forest','mlp','tabnet')
SEED=42


@dataclass
class FittedModel:
    name: str
    prep: object
    clf: object
    history: dict
    settings: dict
    train_rows: int
    fit_seconds: float

    def predict_proba(self,X):
        if len(X)==0:
            return np.empty((0,2))
        transformed=np.asarray(self.prep.transform(X),dtype=np.float32)
        with threadpool_limits(limits=2):
            p=np.asarray(self.clf.predict_proba(transformed),dtype=float)
        if p.shape!=(len(X),2) or not np.isfinite(p).all():
            raise ValueError(f'{self.name}: invalid binary probabilities')
        return p

    def predict(self,X):
        return (self.predict_proba(X)[:,1]>=.5).astype(int)


def fit_model(name,dataset,train_idx,val_idx,quick=False):
    if name not in MODEL_NAMES:
        raise ValueError(f'unknown model {name}')
    start=perf_counter()
    prep=make_preprocessor(dataset,scale_numeric=name not in ('dummy_prior','random_forest'))
    X=dataset.features(); y=dataset.labels()
    xt=np.asarray(prep.fit_transform(X.loc[train_idx]),dtype=np.float32)
    xv=np.asarray(prep.transform(X.loc[val_idx]),dtype=np.float32)
    yt=y.loc[train_idx].to_numpy(); yv=y.loc[val_idx].to_numpy()
    history={}; settings={'seed':SEED,'class_weight':None,'device':'cpu','quick':quick}
    with threadpool_limits(limits=2):
        if name=='dummy_prior':
            clf=DummyClassifier(strategy='prior').fit(xt,yt)
        elif name=='logistic_regression':
            clf=LogisticRegression(max_iter=2000,random_state=SEED).fit(xt,yt)
            settings.update(C=1.0,max_iter=2000)
        elif name=='random_forest':
            trees=40 if quick else 300
            clf=RandomForestClassifier(n_estimators=trees,min_samples_leaf=5,n_jobs=2,random_state=SEED).fit(xt,yt)
            settings.update(n_estimators=trees,min_samples_leaf=5)
        elif name=='mlp':
            clf=MLPClassifier(hidden_layer_sizes=(64,32),activation='relu',alpha=.001,
                              batch_size=min(128,len(yt)),learning_rate_init=.001,random_state=SEED)
            budget=3 if quick else 120
            best=None; best_auc=-np.inf; stale=0
            history={'loss':[],'validation_auc':[]}
            for epoch in range(budget):
                clf.partial_fit(xt,yt,classes=np.array([0,1]))
                auc=roc_auc_score(yv,clf.predict_proba(xv)[:,1])
                history['loss'].append(float(clf.loss_)); history['validation_auc'].append(float(auc))
                if auc>best_auc+1e-5:
                    best_auc=auc; best=deepcopy(clf); stale=0; best_epoch=epoch+1
                else:
                    stale+=1
                if stale>=15:
                    break
            clf=best
            settings.update(hidden_layers=[64,32],alpha=.001,learning_rate=.001,max_epochs=budget,
                            patience=15,best_epoch=best_epoch,epochs_run=len(history['loss']))
        else:
            try:
                import torch
                from pytorch_tabnet.tab_model import TabNetClassifier
            except ImportError as e:
                raise RuntimeError('TabNet is required: install requirements.txt and CPU torch; model was NOT skipped') from e
            torch.set_num_threads(2)
            torch.manual_seed(SEED)
            budget=3 if quick else 80
            clf=TabNetClassifier(n_d=8,n_a=8,n_steps=3,seed=SEED,device_name='cpu',verbose=0,
                                  optimizer_params={'lr':.02})
            clf.fit(xt,yt,eval_set=[(xv,yv)],eval_name=['validation'],eval_metric=['auc'],
                    max_epochs=budget,patience=12,batch_size=256,virtual_batch_size=64,
                    num_workers=0,drop_last=False)
            history={k:[float(v) for v in values] for k,values in clf.history.history.items()}
            settings.update(n_d=8,n_a=8,n_steps=3,max_epochs=budget,patience=12,
                            best_epoch=int(clf.best_epoch)+1,epochs_run=len(history['loss']))
    return FittedModel(name,prep,clf,history,settings,len(train_idx),perf_counter()-start)
