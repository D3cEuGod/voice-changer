"""Regenerate activation and installed entry points after moving a Windows venv.

Run this with each moved environment's python.exe. Installed libraries and
Python redirectors remain in place; no packages are downloaded or removed.
"""
from importlib.metadata import distributions
import json
from pathlib import Path
import sys
import venv


def main():
    folder=Path(sys.prefix).resolve()
    if not (folder/'pyvenv.cfg').is_file():
        raise RuntimeError('Run with the relocated virtual environment, not system Python')
    builder=venv.EnvBuilder(with_pip=False)
    context=builder.ensure_directories(str(folder))
    builder.create_configuration(context)
    builder.setup_scripts(context)
    from pip._vendor.distlib.scripts import ScriptMaker
    maker=ScriptMaker(None,str(folder/'Scripts'))
    maker.executable=str(folder/'Scripts/python.exe')
    maker.clobber=True
    count=0
    for distribution in distributions():
        for entry in distribution.entry_points:
            if entry.group not in ('console_scripts','gui_scripts'):
                continue
            maker.variants={'','X','X.Y'} if entry.name=='pip' else {''}
            maker.make(f'{entry.name} = {entry.value}',options={'gui':entry.group=='gui_scripts'})
            count+=1
    print(json.dumps({'environment':str(folder),'entry_points_rebuilt':count}),flush=True)


if __name__=='__main__':main()
