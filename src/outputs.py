from dataclasses import dataclass
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT=ROOT/'reports/crm/final'
SMOKE_OUTPUT=ROOT/'reports/crm/smoke'
SIMULATED_OUTPUT=ROOT/'reports/crm/simulated'


@dataclass(frozen=True)
class Outputs:
    root: Path

    def __post_init__(self): object.__setattr__(self,'root',Path(self.root))
    @property
    def data(self): return self.root/'data'
    @property
    def models(self): return self.root/'models'
    @property
    def metrics(self): return self.root/'metrics'
    @property
    def explain(self): return self.root/'explain'
    @property
    def checks(self): return self.root/'checks'
    @property
    def predictions(self): return self.root/'predictions'
    @property
    def figures(self): return self.root/'figures'
    @property
    def manifest(self): return self.root/'run_manifest.json'

    def make(self):
        for path in [self.data,self.models,self.metrics,self.explain,self.checks,self.predictions,self.figures]:
            path.mkdir(parents=True,exist_ok=True)
        return self

    def read_manifest(self):
        return json.loads(self.manifest.read_text()) if self.manifest.exists() else {}

    def write_manifest(self,manifest): write_json(self.manifest,manifest)

    def source_data(self):
        from src.data.crm import CRM_DIR
        value=self.read_manifest().get('data_dir')
        if value is None: return CRM_DIR
        path=Path(value)
        return path if path.is_absolute() else ROOT/path


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2,default=str),encoding='utf-8')
