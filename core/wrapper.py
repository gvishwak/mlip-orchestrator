from genomix import GeneticAlgorithm
from math import isclose
from string import Template

from pymatgen.io.lammps.data import LammpsData
from pymatgen.io.lammps.inputs import LammpsRun
from pymatgen.io.vasp.inputs import Incar, Poscar
from pymatgen.io.vasp.outputs import Vasprun, Xdatcar

# from plot_ins import plot_ga_folder, calculate_metrics_for_ga_generation
from utils import *

import numpy as np
import pandas as pd
import os, time, random, json, shutil, subprocess



def convert_to_nequip(vasp_dir, output_dir):
    """Convert VASP data to NequIP format"""
    system = dpdata.LabeledSystem(vasp_dir, fmt="vasp/outcar")
    system.to("nequip/npz", output_dir)
    with open(output_dir/"config.yaml", "w") as f:
        f.write(Path("templates/mlff/template_nequip.yaml").read_text())



def convert_to_allegro(vasp_dir, output_dir):
    """Convert VASP data to Allegro format"""
    system = dpdata.LabeledSystem(vasp_dir, fmt="vasp/outcar")
    system.to("allegro/npz", output_dir)
    shutil.copy("templates/mlff/template_allegro.yaml", output_dir)



def run_lammps_md(structure, potential, template, output_dir):
    """Run LAMMPS simulation with given potential"""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Write LAMMPS inputs
    lammps_data = LammpsData.from_structure(structure)
    lammps_data.write_file(output_dir/"structure.lmp")
    
    with open(template) as f:
        script = f.read().format(
            potential_path=potential,
            structure_path=output_dir/"structure.lmp"
        )
    
    with open(output_dir/"in.lammps", "w") as f:
        f.write(script)
    
    # Execute simulation
    os.system(f"lmp -in {output_dir/'in.lammps'}")


