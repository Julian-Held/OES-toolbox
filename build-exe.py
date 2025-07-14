#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pathlib import Path
import nsist
from platform import python_version
import os
import subprocess

from distutils.util import convert_path
import shutil

build_folder = "./build/"
dist_folder = "./dist/"
shutil.rmtree(build_folder)
os.makedirs(build_folder)


req = subprocess.run(['pip-compile', 
                     '-o', os.path.abspath(build_folder+'requirements.txt'), 
                      '--no-annotate', '--no-header'], check=True, text=True)

req = subprocess.run(['pip', 'wheel', '--pre', '--no-deps', '-w '+build_folder,
                      'git+https://www.github.com/fujiisoup/sif_parser']
                     , check=True, text=True)

main_ns = {}
ver_path = convert_path('OES_toolbox/_version.py')
with open(ver_path) as ver_file:
    exec(ver_file.read(), main_ns)
version = main_ns['version']

name = "OES-toolbox"

icon = Path('./icon.ico')
requirements =open(build_folder+'/requirements.txt').read().strip().split('\n')

for idx,element in enumerate(requirements):
    if "sif-parser" in element:
        requirements.pop(idx)
        break
    
tb_wheel_path = dist_folder + "oes_toolbox-" + version + "-py3-none-any.whl"

sp_wheel_path = os.listdir(build_folder)   
for idx,element in enumerate(sp_wheel_path):
    if "sif_parser" in element:
        break
sp_wheel_path = sp_wheel_path[idx] 

builder = nsist.InstallerBuilder(
    appname=name,
    version=version,
    icon=icon,
    shortcuts={
        name: {
            'entry_point': 'launch_shim:main',
            'console': False,
            'icon': icon,
        }
    },
    py_version=python_version(),
    py_bitness=64,
    pypi_wheel_reqs=requirements,
    local_wheels=[tb_wheel_path, build_folder+sp_wheel_path],
    extra_files=[("./LICENSE", '$INSTDIR'),]
)

print("Building windows installer. This will take a long time.")
builder.run()


# Stub interface to allow hatch to run the file without errors when called as 
# a custom build script. Does nothing.
try:
    from hatchling.builders.plugin.interface import BuilderInterface
    class CustomBuilder(BuilderInterface):
        def __init__(self, root='', plugin_manager=None, config=None, 
                     metadata=None, app=None):
            self._BuilderInterface__metadata=metadata
            self._BuilderInterface__config=config
            self._BuilderInterface__root=root
            self._BuilderInterface__build_config=None
            self._BuilderInterface__target_config=None
            self._BuilderInterface__plugin_manager=None
            self._BuilderInterface__app=None
        def get_version_api(a):
            return []
except:
    pass
