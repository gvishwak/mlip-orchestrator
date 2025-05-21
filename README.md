# MLIP-Orchestrator
<!-- [![APL Machine Learning](https://img.shields.io/badge/APL-Machine%20Learning-blue)](https://pubs.aip.org/aip/apl) -->

A fully modular, end-to-end Python workflow for:

1. **VASP AIMD simulations**
2. **ML force field training** (DeepMD-kit, NequIP, Allegro) with genetic algorithm driven hyperparameter optimization from the [GenomiX](https://github.com/gvishwak/genomix) package
3. **Deployment** of trained models to portable formats
4. **Classical MD simulations with LAMMPS** using the trained ML potentials
5. **Inelastic neutron scattering (INS) spectra** generation via OCLIMAX


---
## Installation


### Dependencies
- Python 3.8+
- [VASP](https://www.vasp.at/) (licensed)
- [LAMMPS](https://www.lammps.org/)
- [DeePMD-kit](https://github.com/deepmodeling/deepmd-kit)
- [NequIP](https://github.com/mir-group/nequip)
- [Allegro](https://github.com/mir-group/allegro)
- [OCLIMAX](https://sites.google.com/site/ornliceman/download)
- [GenomiX](https://github.com/gvishwak/genomix)



### 🛠️ Environment Setup

Create and activate a conda environment:

```bash
conda env create -f environment.yaml
conda activate mlip_workflow

# Install other packages (follow instructions in the respective GitHub repos): deepmd-kit, nequip, allegro, genomix
# Install OCLIMAX (follow instructions at https://sites.google.com/site/ornliceman/download)

git clone https://github.com/gvishwak/mlip-orchestrator.git
```

<!-- 
---
### ⚙️ 3. Directory Structure

```
apl-ml-workflow/
├── utils.py               # Utility functions (VASP, data prep, LAMMPS, INS)
├── wrapper.py             # Workflow orchestration (training, deployment, MD, INS)
├── templates/
│   ├── vasp/              # INCAR, KPOINTS templates
│   ├── mlff/              # template_<model>.yaml for DeepMD, NequIP, Allegro
│   └── lammps/            # LAMMPS input templates
├── extern/genomix/        # GA package
└── README.md
``` -->

---

## Workflow Usage
<!-- 
### ▶️ 4. Workflow Execution

Below is the typical sequence of function calls in a Python script (e.g. `run_all.py`):

```python
from utils import (
    poscar_from_cif, create_potcar, create_kpoints,
    generate_training_data, deploy_trained_model
)
from wrapper import (
    train_mlff_model, run_lammps_md, generate_neutron_scattering_spectra
)

# 1. Prepare VASP input
poscar_from_cif('structure.cif', 'input/')            # or user supplies POSCAR
create_potcar(elements=['H','O'], save_path='input/')
create_kpoints(save_path='input/')

# 2. Run VASP AIMD manually: vasp_std > input/OUTCAR
#    If no INCAR in templates/vasp, will default to NVT 150K

# 3. Generate MLFF training data
generate_training_data(
    model_type='deepmd', base_path='.', nsteps=10000,
    validation_n_sets=5, structure_name='ice_pure'
)

# 4. GA‑driven training (gen_no from genomix GA)
train_mlff_model(
    base_path='.', scratch_path='./scratch', gen_no=1,
    cluster='local', nersc_gpu_nodes=1,
    model_type='nequip', structure_name='ice_pure'
)

# 5. Deploy models
deploy_trained_model(scratch_path='./scratch', gen_no=1, model_type='nequip')

# 6. LAMMPS MD with ML potential
run_lammps_md(
    base_path='.', scratch_path='./scratch', gen_no=1,
    lammps_T=150, vasp_timestep=2.0, model_type='nequip', structure_name='ice_pure'
)

# 7. INS spectra
generate_neutron_scattering_spectra(
    MD_code='lammps', base_path='.', scratch_path='./scratch',
    gen_no=1, trj_file_name='ice_pure_lammps_2.dump', timestep_fs=2.0
)
```

Individual steps can be run separately for debugging or parameter scans.

--- -->


<!-- 
### 1. VASP AIMD Setup
```
from workflow import MLFFWorkflow

wf = MLFFWorkflow("structure.cif")
wf.setup_vasp_simulation(temp=300, steps=5000)
```

### 2. Generate Training Data
```
wf.generate_ml_data("vasp_aimd/")
```

### 3. Train MLFF Models
```
ga_config = {
    "population_size": 30,
    "generations": 20,
    "mutation_rate": 0.1
}
best_models = wf.train_models(ga_config)
```

### 4. Deploy Models
```
wf.deploy_models({
    "deepmd": "deepmd_model.pb",
    "nequip": "nequip_model.pth", 
    "allegro": "allegro_model.pth"
})
```

### 5. LAMMPS Validation
```
wf.run_lammps({
    "deepmd": "deployed/deepmd.pb",
    "nequip": "deployed/nequip.pth",
    "allegro": "deployed/allegro.pth"
})
```

### 6. Analysis
```
wf.analyze_results()
```

## Workflow Diagram -->






## Citation
Please cite our work and the supporting packages:
```
@article{vishwakarma2019towards,
  title={Towards autonomous machine learning in chemistry via evolutionary algorithms},
  author={Vishwakarma, Gaurav and Haghighatlari, Mojtaba and Hachmann, Johannes},
  doi={10.26434/chemrxiv.9782387.v1},
  year={2019}
}
```

## Support
For questions or issues, please open a GitHub issue or contact [your.email@institution.edu](mailto:your.email@institution.edu).


### 🤝 Contributing

Please submit issues or pull requests to add new features, fix bugs, or improve documentation.

---

### 📜 License

This work is released under the MIT License.


