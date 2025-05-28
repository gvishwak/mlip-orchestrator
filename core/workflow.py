from dpdata import LabeledSystem
from genomix import GeneticAlgorithm
from math import isclose
from pathlib import Path
from pymatgen.core import Structure
from pymatgen.io.lammps.data import LammpsData
from pymatgen.io.lammps.inputs import LammpsRun
from pymatgen.io.vasp.inputs import Incar, Poscar, Kpoints
from pymatgen.io.vasp.outputs import Vasprun, Xdatcar
from string import Template
from typing import Callable, Dict, Tuple, Any
from utils import *
from wrapper import *

import numpy as np
import pandas as pd
import os, time, random, json, shutil, subprocess, shlex



class VASP_AIMD:
	"""Class for setting up and managing VASP AIMD simulations"""
	
	def __init__(self, structure_path: str):
		self.structure = self._load_structure(structure_path)
		self._validate_structure()


	def _load_structure(self, path: str) -> Structure:
		"""Load structure from CIF, POSCAR/CONTCAR, CHGCAR, LOCPOT, vasprun.xml, CSSR, Netcdf and pymatgen's JSON-serialized structures."""
		path = Path(path)
		if not path.exists():
			raise FileNotFoundError(f"Structure file {path} not found")
			
		return Structure.from_file(path)


	def _validate_structure(self):
		"""Validate structure input"""
		if len(self.structure) == 0:
			raise ValueError("Invalid structure - contains no atoms")
		if not self.structure.is_ordered:
			raise ValueError("Structure contains partial occupancies - not supported")


	def _create_incar(self, path: Path, incar_template_path: str|bool = False, incar_dict: dict|bool = False):
		"""Create INCAR with validation. Sample INCAR in "templates/vasp/INCAR".
		Sample INCAR dict: {
							"IBRION": 0, "NSW": steps, "POTIM": timestep,
							"TEBEG": temp, "TEEND": temp, "SMASS": 0.5
						}
		"""
		if not incar_dict and not incar_template_path:
			raise ValueError('Either provide the path to an existing INCAR file or a dictionary containing INCAR parameters.')

		incar_path = path/"INCAR"

		if incar_template_path:
			if Path(incar_template_path).exists():
				shutil.copy(incar_template_path, incar_path)
			else:
				raise FileNotFoundError()
		else:
			Incar.from_dict(incar_dict).write_file(incar_path)


	def _create_potcar(self, output_dir: Path, potcar_paths: list = []):
		"""Create INCAR with validation. Sample INCAR in "templates/vasp/INCAR".
		Sample INCAR dict: {
							"IBRION": 0, "NSW": steps, "POTIM": timestep,
							"TEBEG": temp, "TEEND": temp, "SMASS": 0.5
						}
		"""
		if not potcar_paths:
			raise ValueError('Provide a list of paths of POTCAR files for each element in the system.')

		if not all([Path(f).exists() for f in potcar_paths]):
			raise FileNotFoundError()

		potcar_list = [str(PotcarSingle.from_file(filename=f)).strip() for f in potcar_paths]

		Path(output_dir/"POTCAR").write_text('\n\n'.join(potcar_list))


	def setup_simulation(
		self, 
		output_dir: str = "vasp_aimd",
		potcar_paths: list = [],
		incar_template_path: str|bool = False,
		incar_dict: dict|bool = False
	) -> Path:
		"""Configure VASP AIMD simulation with validation"""
		vasp_dir = Path(output_dir)
		vasp_dir.mkdir(exist_ok=True)

		# Create input files with validation
		self._create_incar(vasp_dir, incar_template_path=incar_template_path, incar_dict=incar_dict)
		self._create_potcar(vasp_dir, potcar_paths=potcar_paths)
		Poscar(self.structure).write_file(vasp_dir/"POSCAR")
		Kpoints.gamma_automatic(kpts=(1, 1, 1), shift=(0, 0, 0)).write_file(vasp_dir/"KPOINTS")

		return vasp_dir.resolve()


	def generate_numpy_training_data(self, vasp_dir: Path, set_size: int = 10000) -> Dict[str, Path]:
		"""Convert AIMD data to '.npy' format for training MLIPs"""
		if not (vasp_dir/"OUTCAR").exists():
			raise FileNotFoundError("OUTCAR not found in VASP directory")

		output_dir = vasp_dir.parent/"training_data"/"deepmd_data"
		# LabeledSystem(file_name=str((vasp_dir/"OUTCAR").resolve()), fmt="vasp/outcar").to_deepmd_raw(output_dir)
		LabeledSystem(file_name=str((vasp_dir/"OUTCAR").resolve()), fmt="vasp/outcar").to_deepmd_npy(file_name=output_dir, set_size=set_size, prec=np.float64)

		return output_dir



class MLFF_Trainer:
	"""Class for MLFF training with genetic algorithm optimization"""

	def __init__(self, data_paths: Dict[str, Path]):
		self.data_paths = self._validate_data_paths(data_paths)
		self.ga_config = None
		self.performance_func = self.calculate_spectral_overlap


	def _validate_paths(self, paths: Dict[str, Path], path_type: str) -> Dict[str, Path]:
		"""Validate training data paths"""
		valid_formats = ["deepmd", "nequip", "allegro"]

		for fmt, path in paths.items():
			if fmt not in valid_formats:
				raise ValueError(f"Invalid format {fmt}, must be one of {valid_formats}")
			if not path.exists():
				raise FileNotFoundError(f"Training data path {path} not found")

			if path_type == 'data':
				if fmt in ["nequip", "allegro"] and path.suffix != '.npz':
					raise ValueError("Nequip and Allegro require .npz files. Refer to their documentation on proper file generation.")

			elif path_type == 'mlip_input_file':
				if fmt in ["nequip", "allegro"] and path.suffix not in ['.yaml', '.yml']:
					raise ValueError("Nequip and Allegro require .yml input files. Refer to their documentation on proper file generation.")
				if fmt == 'deepmd' and path.suffix != '.json':
					raise ValueError("DeePMDKit requires .json input files. Refer to their documentation on proper file generation.")

			elif path_type == 'deploy':
				if fmt in ['nequip', 'allegro']:
					if 'best_model.pth' not in path.iterdir():
						raise FileNotFoundError(f"Trained potential file not found at {path}.")
				elif fmt == 'deepmd':
					if not any([True for i in path.iterdir() if 'model.ckpt-' in i]):
						raise FileNotFoundError(f"Trained potential file not found at {path}.")

		return paths


	def configure_ga(self,
		space: dict,
		loss_func: Callable,
		loss_func_args: dict = None,
		fitness: tuple = ("Min", ),
		population_size: int = 20,
		generations: int = 100,
		crossover_size: int = 12,
		crossover_type: str = "Blend",
		mutation_size: int = 8,
		mutation_fraction: float = 0.6,
		algorithm: int = 1,
	):
		"""Configure genetic algorithm parameters with validation"""
		if population_size < 2:
			raise ValueError("Population size must be ≥ 2")
		if generations < 1:
			raise ValueError("Number of generations must be ≥ 1")
		if not 0 < mutation_fraction < 1:
			raise ValueError("Mutation rate must be between 0 and 1")

		self.ga_config = {
			'space': space,
			'objective_function': loss_func,
			'objective_function_params': loss_func_args,
			'fitness': fitness,
			"population_size": population_size,
			"generations": generations,
			'crossover_size': crossover_size,
			"crossover_type": crossover_type,
			'mutation_size': mutation_size,
			"mutation_fraction": mutation_fraction,
			'algorithm': algorithm,
		}


	def generate_population(self, fitness_values_dict: dict, batch_mode: bool = True) -> list | Tuple[pd.DataFrame, dict]:
		"""Train models using genetic algorithm"""
		if not self.ga_config:
			raise ValueError("GA configuration not set - call configure_ga() first")

		ga = GeneticAlgorithm(**self.ga_config)
		if batch_mode:
			ga_population = ga.search(batch_mode=batch_mode, fitness_dict=fitness_values_dict)

		return ga_population


	def generate_mlip_input_files(self, mlip_input_file_paths: Dict[str, Path], input_parameters: Dict[str, Dict], output_dir: Dict[str, Path] = {}):
		output_dir = Path.cwd() if not output_dir else output_dir
		mlip_input_file_paths = self._validate_paths(mlip_input_file_paths, 'mlip_input_file')
		mlip_template = {k: Template(Path(mlip_input_file_paths[k]).read_text()).substitute(input_parameters[k]) for k in mlip_input_file_paths}
		for model_name, f_text in mlip_template.items():
			if model_name == 'deepmd':
				json_file = Path(output_dir / model_name / 'input.json').open('w')
				json.dump(json.loads(f_text), json_file, indent=4)
			else:
				Path(output_dir / model_name / f'{model_name}.yaml').write_text(f_text)


	def deploy_models(self, trained_models: Dict[str, Path]):
		"""Convert trained models to production formats"""
		trained_models = self._validate_paths(trained_models, 'deploy')
		for fmt, path in trained_models.items():
			if fmt in ['nequip', 'allegro']:
				cmd = f'nequip-deploy build --train-dir {path} {fmt}.pth'
			elif fmt == 'deepmd':
				cmd = f'dp freeze -o {fmt}.pb'

			r = subprocess.run(shlex.split(cmd), capture_output=True, text=True)




class LAMMPS_MD:
	"""Class for LAMMPS simulations and results analysis"""

	def __init__(self, structure: Structure):
		self.structure = structure
		self._validate_structure()

	def _validate_structure(self):
		"""Validate input structure"""
		if not isinstance(self.structure, Structure):
			raise TypeError("Input must be pymatgen Structure object")
		if len(self.structure) == 0:
			raise ValueError("Structure contains no atoms")


	def run_simulation(self, potential_path: Path, template: str = "nvt", output_dir: Path = Path("lammps_sim")) -> Path:
		"""Run LAMMPS simulation with validation"""
		if not potential_path.exists():
			raise FileNotFoundError(f"Potential file {potential_path} not found")
			
		output_dir.mkdir(exist_ok=True)
		self._write_lammps_inputs(potential_path, template, output_dir)
		self._execute_lammps(output_dir)
		return output_dir


	def _write_lammps_inputs(self, potential: Path, template: str, output_dir: Path):
		"""Generate LAMMPS input files"""
		lammps_data = LammpsData.from_structure(self.structure)
		lammps_data.write_file(output_dir/"structure.lmp")
		
		template_path = f"templates/lammps/{template}.lmp"
		if not Path(template_path).exists():
			raise FileNotFoundError(f"Template {template_path} not found")
			
		with open(template_path) as f:
			script = f.read().format(
				potential_path=potential,
				structure_path=output_dir/"structure.lmp"
			)
			
		with open(output_dir/"in.lammps", "w") as f:
			f.write(script)


	def _execute_lammps(self, output_dir: Path):
		"""Execute LAMMPS simulation"""
		try:
			subprocess.run(
				["lmp", "-in", "in.lammps"],
				cwd=output_dir,
				check=True,
				capture_output=True
			)
		except subprocess.CalledProcessError as e:
			raise RuntimeError(f"LAMMPS execution failed: {e.stderr.decode()}")





