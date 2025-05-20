import shutil
import os, json, random, warnings, ruamel.yaml, re
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

import pymatgen.core as mg
from pymatgen.core.periodic_table import Element
from pymatgen.io.cif import CifParser, CifWriter
from pymatgen.io.lammps.data import LammpsData
from pymatgen.io.lammps.inputs import LammpsRun
from pymatgen.io.vasp.inputs import Incar, Poscar, Potcar, PotcarSingle, Kpoints
from pymatgen.io.vasp.outputs import Vasprun, Xdatcar, Oszicar

from math import isclose
from pathlib import Path
from ruamel.yaml import YAML
from scipy.interpolate import griddata
from string import Template


import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')



def create_incar(
    save_path,
    template_incar_path=None,
    temperature=150,
    nsw=10000,
    potim=2.0,
    encut=520,
    ismear=0,
    sigma=0.05,
    isif=2,
    ibrion=0,
    smass=-3
):
    """
    Write an INCAR into save_path.  If template_incar_path exists, copy it;
    otherwise generate a basic NVT AIMD INCAR at `temperature` K.
    """
    dest = os.path.join(save_path, 'INCAR')
    if template_incar_path and os.path.exists(template_incar_path):
        shutil.copy(template_incar_path, dest)
    else:
        incar = Incar({
            'SYSTEM':        f'NVT AIMD at {temperature}K',
            'ENCUT':         encut,
            'ISMEAR':        ismear,
            'SIGMA':         sigma,
            'IBRION':        ibrion,
            'SMASS':         smass,
            'POTIM':         potim,
            'NSW':           nsw,
            'TEBEG':         temperature,
            'TEEND':         temperature,
            'ISIF':          isif,
            'PREC':          'Normal',
        })
        incar.write_file(dest)


def deploy_trained_model(scratch_path, gen_no, model_type):
    """
    Wrapper around the GA-generation deploy step.
    """
    from wrapper import deploy_trained_models
    # the wrapper expects (scratch_path, gen_no, model_type)
    deploy_trained_models(scratch_path, gen_no, model_type)



