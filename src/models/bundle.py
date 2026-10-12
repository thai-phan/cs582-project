from dataclasses import dataclass,field
import joblib

TRAINED_FILE='trained_models.joblib'
SCORING_FILE='model_bundle.joblib'
WARNING='Trusted local artifact only; never unpickle an untrusted file'


@dataclass
class Trained:
    models: dict
    thresholds: dict
    selected: str
    calibrated: object
    decision_threshold: float
    reference: dict
    features: list
    validation_rows: list=field(default_factory=list)


def save(out,trained):
    joblib.dump(trained,out.models/TRAINED_FILE,compress=3)
    joblib.dump({'model':trained.calibrated,'features':trained.features,'reference':trained.reference,
                 'decision_threshold':trained.decision_threshold,'warning':WARNING},out.models/SCORING_FILE,compress=3)


def load_trained(out):
    trained=joblib.load(out.models/TRAINED_FILE)
    if not isinstance(trained,Trained): raise TypeError(f'{TRAINED_FILE} is not a training artifact of this project')
    return trained


def load_scoring(path):
    bundle=joblib.load(path)
    missing={'model','features','reference','decision_threshold'}-set(bundle)
    if missing: raise ValueError(f'scoring bundle lacks {sorted(missing)}')
    return bundle
