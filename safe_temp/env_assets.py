# safe_temp/env_assets.py
from typing import List, Dict
from packaging.version import parse
from platform import system

so = system().upper()
WIN = "" if so == "LINUX" else "WINDOWS"
# Flags de estadp
greater = "greater"
smaller = "smaller" # menor que establecido
same = "same"   # misma version
between = "between" # entre 2 versiones < lib o lib >
especific_versions = "especific_versions" # Identica en al menos 1 de las versiones requeridas

major_versions = "{major:02d}"
minor_versions = "{minor:02d}"
patch_versions = "{patch:02d}"

python_base = "3.12"
python_patch_versions = [5, 8, 13, 15]

conda_base = "2"
conda_versions: List[int] = [6, 7] # 26.7.2; la version a la fecha de desarrollo '17-11-2026' que conda permite antes de sus implementaciones de seguridad. ver + 27: 
conda_minor_versions = 2

libs_list: List[str] =         ["cmake", "conda",     "cython",  "ninja",  "numpy",   "opencv",   "python"]   # ordenados alfabeticamente
some_version_list: List[str] = ["3.25.0", conda_base, "3.3.0",   "1.6.0",  "1.26.4",  "4.11.0",    python_base]
requeriments_list: List[str] = [greater,  smaller,     same,      greater,  same,      same,     especific_versions]

libs_ver = [f"{libs_list[i]}-" if i in (0, 2) else f"{val}-{some_version_list[i]}" for i, val in enumerate(libs_list)]

def get_libs_versions(version_base: str, **versions: object) -> str:
    return f"{version_base}".format(**versions)

def version_call(lib_found: str, requirements_idx: int) -> str:
    if requirements_idx != 1:
        requirement = some_version_list[requirements_idx]
    else:
        requirement = _get_requirement_val(requirements_idx)

    comparation = requeriments_list[requirements_idx]

    if same == comparation:
        return same if same_comparation(lib_found, requirement) else ""

    elif greater == comparation:
        return greater if (parse(lib_found) >= parse(requirement)) else ""

    elif smaller == comparation:
        return smaller if (parse(requirement) >= parse(lib_found)) else ""

    elif between == comparation:
        return between if (parse(lib_found) > parse(requirement) > parse(lib_found)) else ""
    
    else:
        python_template =f"{python_base}" + "." + patch_versions
        valid_comparation = True
        reformat = False
        for i, patches in enumerate(python_patch_versions):
            if patches < 10:
                reformat = True
                patches = python_patch_versions[i] * 10
                
            versions_dict = {"patch": patches}
            n_version = get_libs_versions(python_template, **versions_dict)
            if reformat:
                n_version = n_version[:-1]

            if same_comparation(lib_found, n_version):
                valid_comparation = True
                break
            else:
                reformat = False
                valid_comparation = False

        return especific_versions if valid_comparation else ""

def _get_requirement_val(idx: int):
    if conda_base != some_version_list[idx]:
        return ""

    versions_dict: Dict[str, int] = {
        "minor": conda_versions[1],
        "patch": conda_minor_versions
    }
    template = f"{conda_base}{7}" + "." + minor_versions + "." + patch_versions
    return get_libs_versions(template, **versions_dict)

def same_comparation(lib_found: str, exact_version: str) -> bool:
    return exact_version == lib_found
