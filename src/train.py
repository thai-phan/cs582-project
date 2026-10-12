import argparse
from datetime import datetime,timezone
import hashlib
from pathlib import Path
from importlib.metadata import version
import platform
import subprocess
import pandas as pd
from src.data.crm import build,CRM_DIR
from src.data.split import asof_split
from src.evaluation.metrics import metrics,choose_threshold,select_model
from src.explain.reference import make_reference
from src.models import bundle
from src.models.calibration import CalibratedModel
from src.models.zoo import MODEL_NAMES,fit_model
from src.outputs import ROOT,DEFAULT_OUTPUT,SMOKE_OUTPUT,SIMULATED_OUTPUT,Outputs,write_json

DEPENDENCIES=['numpy','pandas','scipy','scikit-learn','torch','pytorch-tabnet','shap','matplotlib','joblib']


def describe_run(manifest,dataset,split,data_dir):
    y=dataset.labels()
    manifest['features']=dataset.feature_columns
    manifest['data_dir']=str(data_dir.relative_to(ROOT)) if data_dir.is_relative_to(ROOT) else str(data_dir)
    inputs=[data_dir/f'{name}.csv' for name in ['sales_pipeline','accounts','products','sales_teams','data_dictionary']]
    manifest['raw_sha256']={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs if path.exists()}
    manifest['split']={'method':'engagement date groups + outcome availability purge',
        'validation_start':str(split.validation_start.date()),'test_start':str(split.test_start.date()),
        'train_rows':len(split.train),'validation_rows':len(split.validation),'test_rows':len(split.test),
        'purged_rows':int(split.manifest.role.str.startswith('purged').sum()),
        'train_win_rate':float(y.loc[split.train].mean()),'validation_win_rate':float(y.loc[split.validation].mean()),
        'test_win_rate':float(y.loc[split.test].mean())}
    manifest['dependencies']={name:version(name) for name in DEPENDENCIES}
    git=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,text=True,capture_output=True)
    manifest['base_commit']=git.stdout.strip() if git.returncode==0 else 'not a git checkout'
    manifest['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'src').rglob('*.py'))}


def default_output(data_dir=CRM_DIR,quick=False):
    if Path(data_dir).resolve()==CRM_DIR.resolve(): return SMOKE_OUTPUT if quick else DEFAULT_OUTPUT
    return SIMULATED_OUTPUT/('smoke' if quick else 'final')


def start(output_dir=DEFAULT_OUTPUT,quick=False,data_dir=CRM_DIR,dataset=None,split=None):
    data_dir=Path(data_dir).resolve()
    if data_dir!=CRM_DIR.resolve() and Path(output_dir).resolve()==DEFAULT_OUTPUT.resolve():
        raise ValueError(f'{DEFAULT_OUTPUT} holds only results on data/crm; choose another --output for {data_dir}')
    out=Outputs(output_dir).make()
    manifest={'status':'running','mode':'SMOKE_TEST_NOT_FINAL' if quick else 'full',
              'started_utc':datetime.now(timezone.utc).isoformat(),'seed':42,'python':platform.python_version(),
              'platform':platform.platform(),'machine':platform.machine(),
              'run_location':'local CPU execution; see platform and machine'}
    out.write_manifest(manifest)
    dataset=dataset or build(data_dir); split=split or asof_split(dataset)
    split.manifest.to_csv(out.data/'split_manifest.csv',index=False)
    write_json(out.data/'data_quality.json',dataset.report)
    describe_run(manifest,dataset,split,data_dir)
    return out,manifest,dataset,split


def fit_one(name,dataset,split,quick=False):
    X=dataset.features(); xv,yv=X.loc[split.validation],dataset.labels().loc[split.validation]
    model=fit_model(name,dataset,split.train,split.validation,quick=quick)
    pv=model.predict_proba(xv)[:,1]; threshold=choose_threshold(yv,pv)
    return model,threshold,{'model':name,**metrics(yv,pv,threshold)}


def finish(out,manifest,dataset,split,fitted):
    if list(fitted)!=list(MODEL_NAMES): raise ValueError(f'fit all of {MODEL_NAMES} in order before selecting; got {list(fitted)}')
    X=dataset.features(); xv,yv=X.loc[split.validation],dataset.labels().loc[split.validation]
    models={name:f[0] for name,f in fitted.items()}; thresholds={name:f[1] for name,f in fitted.items()}
    rows=[f[2] for f in fitted.values()]
    histories={name:{'settings':m.settings,'history':m.history,'fit_seconds':m.fit_seconds} for name,m in models.items()}
    selected=select_model(rows)
    calibrated=CalibratedModel.fit(models[selected],xv,yv)
    reference=make_reference(X.loc[split.train],dataset)
    trained=bundle.Trained(models=models,thresholds=thresholds,selected=selected,calibrated=calibrated,
        decision_threshold=float(calibrated.transform(thresholds[selected])),reference=reference,
        features=dataset.feature_columns,validation_rows=rows)
    pd.DataFrame(rows).to_csv(out.metrics/'validation_metrics.csv',index=False)
    write_json(out.models/'model_training.json',histories)
    write_json(out.models/'explanation_reference.json',reference)
    bundle.save(out,trained)
    manifest.update(selected_model=selected,selection_rule='validation ROC-AUC, then lower Brier, then model name',
        calibration={'method':'positive-slope sigmoid on validation logit scores; base remains frozen',
                     'slope':calibrated.slope,'intercept':calibrated.intercept},
        raw_decision_threshold=thresholds[selected],calibrated_decision_threshold=trained.decision_threshold,
        threshold_rule='validation macro-F1; 0.10..0.90 by 0.01; ties closest to 0.50',
        priority={'medium_min':.4,'high_min':.7,'status':'unvalidated heuristic, not a business policy'},
        status='trained',stages={'train':datetime.now(timezone.utc).isoformat()})
    out.write_manifest(manifest)
    print(f'TRAINED: selected={selected}; models saved in {out.models}',flush=True)
    return trained


def run(output_dir=DEFAULT_OUTPUT,quick=False,dataset=None,split=None,data_dir=CRM_DIR):
    out,manifest,dataset,split=start(output_dir,quick,data_dir,dataset,split)
    fitted={}
    for name in MODEL_NAMES:
        print(f'Training {name} on CPU...',flush=True)
        fitted[name]=fit_one(name,dataset,split,quick=quick)
    trained=finish(out,manifest,dataset,split,fitted)
    return manifest,trained,dataset,split


def main():
    parser=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--quick',action='store_true',help='All five models, tiny budgets; not reportable results')
    parser.add_argument('--output',type=Path,default=None)
    parser.add_argument('--data-dir',type=Path,default=CRM_DIR,help='Folder with sales_pipeline/accounts/products/sales_teams CSVs')
    args=parser.parse_args()
    run(args.output or default_output(args.data_dir,args.quick),quick=args.quick,data_dir=args.data_dir)


if __name__=='__main__': main()
