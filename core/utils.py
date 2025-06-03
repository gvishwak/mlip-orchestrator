from pathlib import Path
from pymatgen.core import Structure



def load_structure(path: str) -> Structure:
    """Load structure from CIF, POSCAR/CONTCAR, CHGCAR, LOCPOT, vasprun.xml, CSSR, Netcdf and pymatgen's JSON-serialized structures."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Structure file {path} not found")

    structure = Structure.from_file(path)

    if len(structure) == 0:
        raise ValueError("Invalid structure - contains no atoms")
    if not structure.is_ordered:
        raise ValueError("Structure contains partial occupancies - not supported")

    return structure



