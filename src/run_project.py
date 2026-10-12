import argparse
from datetime import datetime,timezone
from pathlib import Path
import time
from src import evaluate,predict,train
from src.outputs import DEFAULT_OUTPUT,SMOKE_OUTPUT,Outputs


def run(output_dir=DEFAULT_OUTPUT,quick=False):
    start=time.perf_counter(); out=Outputs(output_dir)
    try:
        _,trained,dataset,split=train.run(out.root,quick=quick)
        evaluate.run(out.root,trained=trained,dataset=dataset,split=split)
        predict.run(out.root,dataset=dataset)
        manifest=out.read_manifest()
        manifest.update(status='complete',completed_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.perf_counter()-start)
        out.write_manifest(manifest)
        print(f"COMPLETE: {out.root}; selected={manifest['selected_model']}; scored={manifest['scored_open_rows']}",flush=True)
        return manifest
    except Exception as exc:
        manifest=out.read_manifest(); manifest.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        if out.root.exists(): out.write_manifest(manifest)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--quick',action='store_true',help='All five models, tiny budgets; not reportable results')
    parser.add_argument('--output',type=Path,default=None)
    args=parser.parse_args()
    run(args.output or (SMOKE_OUTPUT if args.quick else DEFAULT_OUTPUT),quick=args.quick)


if __name__=='__main__': main()
