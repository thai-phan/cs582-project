import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

ROOT=Path(__file__).resolve().parents[2]
PATHS=['.gitignore','AGENTS.md','README.md','PLAN.md','CS582_Group2_Project_Proposal.md','requirements.txt',
       'src','tests','scripts','docs','reports/crm']


def git(*args,env=None):
    return subprocess.check_output(['git',*args],cwd=ROOT,env=env)


def package(destination):
    destination=Path(destination).resolve(); destination.mkdir(parents=True,exist_ok=True)
    if destination.is_relative_to(ROOT):
        raise ValueError('Write handoff outside the repository to avoid packaging itself')
    base=git('rev-parse','HEAD').decode().strip()
    actual_index=Path(git('rev-parse','--git-path','index').decode().strip())
    if not actual_index.is_absolute(): actual_index=ROOT/actual_index
    before=hashlib.sha256(actual_index.read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory(prefix='crm-package-') as temp:
        env=dict(os.environ,GIT_INDEX_FILE=str(Path(temp)/'index'))
        git('read-tree',base,env=env)
        git('add','-A','--',*PATHS,env=env)
        changed=git('diff','--cached','--name-only',base,env=env).decode().splitlines()
        if any(path.startswith('data/') for path in changed):
            raise AssertionError('Handoff must not modify any raw data')
        tree=git('write-tree',env=env).decode().strip()
        archive=git('archive','--format=zip','--prefix=crm-sales-opportunities/',tree)
        (destination/'CRM_CS582_Complete.zip').write_bytes(archive)
        patch=git('diff','--cached','--binary',base,env=env)
        (destination/'crm-completion.patch').write_bytes(patch)
    if hashlib.sha256(actual_index.read_bytes()).hexdigest()!=before:
        raise AssertionError('Real Git index unexpectedly changed')
    hashes={name:hashlib.sha256((destination/name).read_bytes()).hexdigest() for name in ['CRM_CS582_Complete.zip','crm-completion.patch']}
    manifest={'base_commit':base,'tree':tree,'changed_files':changed,'sha256':hashes,
              'git_status':'no commit, branch change, push or PR; real index unchanged'}
    (destination/'HANDOFF.json').write_text(json.dumps(manifest,indent=2))
    (destination/'HANDOFF.md').write_text(f'''# CRM handoff

Base commit: `{base}`. No commit or push was performed.

`CRM_CS582_Complete.zip` contains the source, raw CRM inputs and current results. Unzip it, then follow README.md (`python scripts/tools/setup_cpu.py`, then `.venv-crm/bin/python -m src.run_project`).

For GitHub: keep the existing repository. Use a fresh feature branch. Either copy the extracted contents into the checkout, or apply `crm-completion.patch` to a clean checkout at the base above. Run `git apply --check crm-completion.patch` before `git apply --index crm-completion.patch`. Do not apply the patch twice. Review and test before commit/push/PR.

The patch changes no raw data. Python environments, Git metadata and credentials are excluded. Hashes and changed-file paths are in HANDOFF.json.

The project has weak predictive discrimination; do not present it as production-ready. Member review and course submission remain external steps.
''',encoding='utf-8')
    print(json.dumps({'files':list(hashes),'base_commit':base,'changed_count':len(changed),'index_unchanged':True},indent=2))
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('destination',type=Path)
    package(parser.parse_args().destination)
