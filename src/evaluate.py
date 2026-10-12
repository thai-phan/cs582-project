import argparse
from dataclasses import dataclass,field
from datetime import datetime,timezone
from pathlib import Path
import pandas as pd
from src.data.crm import build
from src.data.split import asof_split
from src.evaluation import audits,figures,novelty_checks
from src.evaluation.metrics import metrics
from src.explain import importance
from src.models import bundle
from src.outputs import DEFAULT_OUTPUT,Outputs

LIMITATIONS=[
    'Previously inspected dataset: exploratory holdout, not an untouched external test.',
    'Outcome-availability purge reduces but cannot remove right-censoring/closed-only selection bias.',
    'Static account/product/team snapshots may not reflect historical values at engagement.',
    'Repeated accounts/agents cross periods: not a new-customer generalization test.',
    'Validation reused for early stopping, selection, threshold and calibration; estimates may be optimistic.',
    'Open scoring is a frozen-model snapshot demo, not an as-of replay of each historical open deal.',
    'Missing-account open deals are out of training support; no validated uplift or causal recommendation.',
    'Probability calibration and priority thresholds need external/prospective validation.']


@dataclass
class Evaluation:
    out: Outputs
    manifest: dict
    trained: bundle.Trained
    dataset: object
    split: object
    quick: bool
    test_rows: list=field(default_factory=list)
    probabilities: dict=field(default_factory=dict)
    calibrated_test: object=None
    test_predictions: object=None

    def xy(self,part):
        idx=getattr(self.split,part)
        return self.dataset.features().loc[idx],self.dataset.labels().loc[idx]


def start(output_dir=DEFAULT_OUTPUT,trained=None,dataset=None,split=None):
    out=Outputs(output_dir).make(); manifest=out.read_manifest()
    if 'selected_model' not in manifest: raise RuntimeError(f'{out.root} has no trained models: run src.train first')
    trained=trained or bundle.load_trained(out)
    dataset=dataset or build(out.source_data()); split=split or asof_split(dataset)
    ev=Evaluation(out,manifest,trained,dataset,split,quick=manifest['mode']!='full')
    figures.eda(dataset,split,out)
    return ev


def score_test(ev):
    t=ev.trained; xt,yt=ev.xy('test'); selected=t.selected
    ev.probabilities={name:model.predict_proba(xt)[:,1] for name,model in t.models.items()}
    ev.test_rows=[{'model':name,**metrics(yt,p,t.thresholds[name])} for name,p in ev.probabilities.items()]
    pd.DataFrame(ev.test_rows).to_csv(ev.out.metrics/'test_metrics.csv',index=False)
    ev.calibrated_test=t.calibrated.predict_proba(xt)[:,1]
    pd.DataFrame([{'variant':'raw',**metrics(yt,ev.probabilities[selected],t.thresholds[selected])},
                  {'variant':'calibrated',**metrics(yt,ev.calibrated_test,t.decision_threshold)}]).to_csv(ev.out.metrics/'calibration_test.csv',index=False)
    ev.test_predictions=ev.dataset.frame.loc[ev.split.test,['opportunity_id','deal_stage','engage_date']].copy()
    for name,p in ev.probabilities.items(): ev.test_predictions['p_'+name]=p
    ev.test_predictions['p_selected_calibrated']=ev.calibrated_test
    ev.test_predictions.to_csv(ev.out.metrics/'test_predictions.csv',index=False)
    return pd.DataFrame(ev.test_rows)


def make_figures(ev):
    t=ev.trained; _,yt=ev.xy('test')
    histories={name:{'settings':m.settings,'history':m.history,'fit_seconds':m.fit_seconds} for name,m in t.models.items()}
    figures.learning_curves(histories,ev.out)
    figures.evaluation_figures(yt,ev.probabilities,t.thresholds,{'raw':ev.probabilities[t.selected],'calibrated':ev.calibrated_test},ev.out)


def explain_models(ev):
    t=ev.trained; xt,yt=ev.xy('test')
    importance.native_importances(t.models,ev.out)
    importance.permutation_importance(t.calibrated,xt,yt,ev.out,quick=ev.quick)
    importance.shap_audit(t.models['random_forest'],xt,ev.test_predictions.opportunity_id,ev.out,quick=ev.quick)


def run_audits(ev):
    t=ev.trained; xv,yv=ev.xy('validation'); _,yt=ev.xy('test')
    audits.leakage_control(ev.dataset,ev.split,next(r for r in ev.test_rows if r['model']==t.selected),ev.out)
    audits.priority_checks(yt,ev.calibrated_test,yv,t.calibrated.predict_proba(xv)[:,1],ev.out)


def run_checks(ev):
    print('Checking split protocols and explanations...',flush=True)
    t=ev.trained
    protocols=novelty_checks.split_protocols(ev.dataset,ev.split,ev.out,quick=ev.quick)
    checks=novelty_checks.explanation_checks(t.models['random_forest'],t.calibrated,ev.dataset,ev.split,t.reference,ev.out,quick=ev.quick)
    ev.manifest['novelty_checks']={'split_protocol_max_auc_gain':float(protocols.delta_vs_asof.max()),
        'rf_explanation_spearman_median':checks['agreement']['spearman_median'],
        'deletion_ratio_top_to_random':checks['deletion']['ratio_top_to_random']}
    return protocols,checks


def finish(ev):
    ev.manifest['limitations']=LIMITATIONS
    ev.manifest.setdefault('stages',{})['evaluate']=datetime.now(timezone.utc).isoformat()
    ev.out.write_manifest(ev.manifest)
    auc=next(r for r in ev.test_rows if r['model']==ev.trained.selected)['roc_auc']
    print(f'EVALUATED: {ev.trained.selected} test ROC-AUC {auc:.4f}',flush=True)
    return ev.manifest


def run(output_dir=DEFAULT_OUTPUT,trained=None,dataset=None,split=None):
    ev=start(output_dir,trained,dataset,split)
    score_test(ev); make_figures(ev); explain_models(ev); run_audits(ev); run_checks(ev)
    return finish(ev)


def main():
    parser=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--output',type=Path,default=None,help='Run folder written by src.train')
    run(parser.parse_args().output or DEFAULT_OUTPUT)


if __name__=='__main__': main()
