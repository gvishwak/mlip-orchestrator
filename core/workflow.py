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
from typing import Callable, Dict, Tuple
from utils import *
from wrapper import *

import numpy as np
import pandas as pd
import os, time, random, json, shutil, subprocess



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


	def _validate_data_paths(self, paths: Dict[str, Path]) -> Dict[str, Path]:
		"""Validate training data paths"""
		valid_formats = ["deepmd", "nequip", "allegro"]
		for fmt, path in paths.items():
			if fmt not in valid_formats:
				raise ValueError(f"Invalid format {fmt}, must be one of {valid_formats}")
			if not path.exists():
				raise FileNotFoundError(f"Training data path {path} not found")
			if fmt in ["nequip", "allegro"] and path.suffix != '.npz':
				raise ValueError("Nequip and Allegro require .npz files. Refer to their documentation on proper file generation.")
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


	def train_models(self, fitness_values_dict: dict, batch_mode: bool = True) -> list | Tuple[pd.DataFrame, dict]:
		"""Train models using genetic algorithm"""
		if not self.ga_config:
			raise ValueError("GA configuration not set - call configure_ga() first")
			
		ga = GeneticAlgorithm(**self.ga_config)
		if batch_mode:
			ga_population = ga.search(batch_mode=batch_mode, fitness_dict=fitness_values_dict)

		return 


	def calculate_spectral_overlap(
		self, 
		model_path: Path, 
		reference_spectrum: Path
	) -> float:
		"""Default model performance function using neutron spectra"""
		# Implementation details would interface with OCLIMAX
		return self._run_oclimax_simulation(model_path, reference_spectrum)
	
	def _run_oclimax_simulation(self, model_path: Path, reference: Path) -> float:
		"""Calculate spectral similarity metric"""
		# Actual implementation would run simulations and calculate overlap
		return np.random.random()  # Placeholder



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
	
	def run_simulation(
		self, 
		potential_path: Path, 
		template: str = "nvt",
		output_dir: Path = Path("lammps_sim")
	) -> Path:
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
	
	def generate_neutron_spectra(
		self, 
		lammps_dir: Path,
		output_dir: Path = Path("spectra")
	) -> pd.DataFrame:
		"""Generate neutron scattering spectra using OCLIMAX"""
		if not (lammps_dir/"log.lammps").exists():
			raise FileNotFoundError("LAMMPS output not found")
			
		output_dir.mkdir(exist_ok=True)
		self._run_oclimax(lammps_dir, output_dir)
		return self._process_spectra(output_dir)
	
	def _run_oclimax(self, lammps_dir: Path, output_dir: Path):
		"""Execute OCLIMAX workflow"""
		# Actual implementation would interface with OCLIMAX
		pass
	
	def _process_spectra(self, output_dir: Path) -> pd.DataFrame:
		"""Process raw spectral data"""
		# Return processed spectral data
		return pd.DataFrame()





























































class MLFFWorkflow:
	def __init__(self, structure_path, output_dir="results"):
		self.structure = self._load_structure(structure_path)
		self.output_dir = Path(output_dir)
		self.output_dir.mkdir(exist_ok=True)


	def _load_structure(self, path):
		"""Load structure from CIF/POSCAR/CONTCAR"""
		path = Path(path)
		if path.suffix == ".cif":
			return CifParser(str(path)).get_structures()[0]
		return Poscar.from_file(str(path)).structure


	def setup_vasp_simulation(self, temp=150, timestep=1, steps=10000):
		"""Configure VASP AIMD simulation"""
		vasp_dir = self.output_dir/"vasp_aimd"
		vasp_dir.mkdir(exist_ok=True)
		
		# Create input files
		incar_path = "templates/vasp/INCAR" if Path("templates/vasp/INCAR").exists() else None
		if incar_path:
			shutil.copy(incar_path, vasp_dir/"INCAR")
		else:
			Incar.from_dict({
				"IBRION": 0, "NSW": steps, "POTIM": timestep,
				"TEBEG": temp, "TEEND": temp, "SMASS": 0.5
			}).write_file(vasp_dir/"INCAR")
			
		Poscar(self.structure).write_file(vasp_dir/"POSCAR")
		Kpoints.gamma_automatic().write_file(vasp_dir/"KPOINTS")
		
		print(f"VASP inputs created in {vasp_dir}. Provide POTCAR to run.")


	def generate_ml_data(self, vasp_run_dir):
		"""Convert AIMD data to ML training formats"""
		LabeledSystem(vasp_run_dir, fmt="vasp/outcar").to(
			"deepmd/npy", self.output_dir/"deepmd_data"
		)
		convert_to_nequip(vasp_run_dir, self.output_dir/"nequip_data")
		convert_to_allegro(vasp_run_dir, self.output_dir/"allegro_data")


	def train_models(self, ga_config):
		"""Train MLFFs with genetic algorithm optimization"""
		ga = GeneticAlgorithm(
			objective_function=self._model_fitness,
			**ga_config
		)
		return ga.search()


	def _model_fitness(self, params):
		"""Evaluate model performance using validation data"""
		# Implementation details in wrapper.py
		pass


	def deploy_models(self, trained_models):
		"""Convert trained models to production formats"""
		for name, path in trained_models.items():
			if "deepmd" in name:
				os.system(f"dp compress -i {path} -o {path}_compressed.pb")
			elif "nequip" in name:
				os.system(f"nequip-deploy convert {path} {path}.pth")
			elif "allegro" in name:
				shutil.copy(path, self.output_dir/"deployed")


	def run_lammps(self, potentials):
		"""Run LAMMPS simulations with trained potentials"""
		for name, potential in potentials.items():
			run_lammps_md(
				structure=self.structure,
				potential=potential,
				template=f"templates/lammps/{name}.lmp",
				output_dir=self.output_dir/f"lammps_{name}"
			)


	def analyze_results(self):
		"""Generate neutron scattering spectra and plots"""
		generate_neutron_scattering_spectra(
			self.output_dir/"vasp_aimd",
			self.output_dir/"lammps_results",
			self.output_dir/"analysis"
		)

