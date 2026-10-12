from pathlib import Path
import subprocess
import sys
import venv

ROOT=Path(__file__).resolve().parents[2]


def main():
    if sys.version_info[:2] != (3,12):
        raise SystemExit('Verified runtime is Python 3.12. Select a Python 3.12 runtime; do not silently change pins.')
    env=ROOT/'.venv-crm'
    venv.create(env,with_pip=True,symlinks=sys.platform!='win32')
    python=env/('Scripts/python.exe' if sys.platform=='win32' else 'bin/python')
    subprocess.run([str(python),'-m','pip','install','torch==2.8.0','--index-url','https://download.pytorch.org/whl/cpu'],check=True)
    subprocess.run([str(python),'-m','pip','install','-r',str(ROOT/'requirements.txt')],check=True)
    subprocess.run([str(python),'-m','pip','check'],check=True)
    print('CPU environment ready:',python)


if __name__=='__main__': main()
