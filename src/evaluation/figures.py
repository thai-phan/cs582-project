import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_auc_score,roc_curve,ConfusionMatrixDisplay,precision_recall_curve,average_precision_score


def save(fig,path):
    fig.tight_layout()
    fig.savefig(path,dpi=160,bbox_inches='tight')
    plt.close(fig)


def eda(dataset,split,out):
    figs=out.figures
    train=dataset.frame.loc[split.train]
    fig,ax=plt.subplots(figsize=(6,4))
    train.deal_stage.value_counts().reindex(['Lost','Won']).plot.bar(ax=ax,color=['#ea9768','#278b9c'],rot=0)
    ax.set(title='Training outcomes (labels available at cutoff)',ylabel='Deals')
    save(fig,figs/'eda_train_outcomes.png')
    fig,ax=plt.subplots(figsize=(7,4))
    train.groupby(train.engage_date.dt.to_period('M')).size().plot.bar(ax=ax,color='#278b9c')
    ax.set(title='Training engagements by month',ylabel='Deals',xlabel='Engagement month')
    save(fig,figs/'eda_train_months.png')
    products=train.groupby('product').is_won.agg(['count','mean']).sort_values('mean')
    products.to_csv(out.data/'eda_product_training.csv')
    fig,ax=plt.subplots(figsize=(7,4))
    products['mean'].plot.barh(ax=ax,color='#278b9c')
    ax.set(title='Training win rate by product (descriptive only)',xlabel='Win rate',xlim=(0,1))
    save(fig,figs/'eda_product_training.png')
    closed=dataset.features().isna().mean()
    opened=dataset.extra['scorable_open_deals'][dataset.feature_columns].isna().mean()
    missing=pd.DataFrame({'closed_missing_fraction':closed,'engaging_missing_fraction':opened})
    missing.to_csv(out.data/'missingness.csv')
    shown=missing.loc[missing.max(axis=1)>0]
    if not len(shown): shown=missing.head(1)
    fig,ax=plt.subplots(figsize=(8,4))
    shown.plot.barh(ax=ax,color=['#278b9c','#ea9768'])
    ax.set(title='Input coverage shifts in open deals',xlabel='Missing fraction',xlim=(0,1))
    ax.legend(['Closed','Engaging'],loc='lower right',fontsize=8)
    save(fig,figs/'eda_missingness.png')


def evaluation_figures(y,probabilities,thresholds,calibrated,out):
    fig,ax=plt.subplots(figsize=(7,5))
    for name,p in probabilities.items():
        fpr,tpr,_=roc_curve(y,p)
        ax.plot(fpr,tpr,label=f'{name}: {roc_auc_score(y,p):.3f}')
        cmfig,cmax=plt.subplots(figsize=(4.5,4))
        ConfusionMatrixDisplay.from_predictions(y,p>=thresholds[name],labels=[0,1],display_labels=['Lost','Won'],ax=cmax,colorbar=False)
        cmax.set_title(name+' (validation threshold)',fontsize=10)
        save(cmfig,out.figures/f'confusion_{name}.png')
    ax.plot([0,1],[0,1],'--',color='gray')
    ax.set(title='Exploratory held-out ROC curves',xlabel='False positive rate',ylabel='True positive rate')
    ax.legend(fontsize=8); save(fig,out.figures/'roc_curves.png')
    fig,ax=plt.subplots(figsize=(7,5))
    for name,p in probabilities.items():
        precision,recall,_=precision_recall_curve(y,p)
        ax.plot(recall,precision,label=f'{name}: AP {average_precision_score(y,p):.3f}')
    ax.axhline(float(np.mean(y)),ls='--',color='gray',label='Test prevalence')
    ax.set(title='Exploratory held-out precision–recall',xlabel='Recall (Won)',ylabel='Precision (Won)',xlim=(0,1),ylim=(0,1.02))
    ax.legend(fontsize=8); save(fig,out.figures/'precision_recall_curves.png')
    fig,ax=plt.subplots(figsize=(6,4))
    for label,p in calibrated.items():
        actual,pred=calibration_curve(y,p,n_bins=6,strategy='quantile')
        ax.plot(pred,actual,'o-',label=label)
    ax.plot([0,1],[0,1],'--',color='gray')
    ax.set(title='Reliability on test (quantile bins)',xlabel='Mean predicted win probability',ylabel='Observed win fraction',xlim=(0,1),ylim=(0,1))
    ax.legend(fontsize=8); save(fig,out.figures/'reliability.png')


def learning_curves(histories,out):
    for name in ['mlp','tabnet']:
        record=histories[name]; history=record['history']
        epochs=np.arange(1,len(history['loss'])+1)
        fig,(loss_ax,auc_ax)=plt.subplots(1,2,figsize=(9,3.5))
        loss_ax.plot(epochs,history['loss'],color='#278b9c')
        loss_ax.set(xlabel='Epoch',ylabel='Training loss')
        auc_ax.plot(epochs,history['validation_auc'],color='#278b9c')
        auc_ax.axvline(record['settings']['best_epoch'],ls='--',color='#db6849',label='Retained epoch')
        auc_ax.set(xlabel='Epoch',ylabel='Validation ROC-AUC'); auc_ax.legend(fontsize=8)
        fig.suptitle(name+' learning history (validation reused for selection)')
        save(fig,out.figures/f'learning_curve_{name}.png')
