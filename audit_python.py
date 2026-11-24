"""
audit_python.py — Auditoría completa del entorno Python.
Sin módulo específico: vuelca TODO lo que el intérprete sabe.
No importa nada del proyecto. Solo lee sys.*, sysconfig, importlib.machinery, os.
"""
import sys
import os
import sysconfig
import importlib.machinery
import importlib.util
import site
import platform

print("=" * 78)
print("AUDITORÍA COMPLETA DEL ENTORNO PYTHON")
print("=" * 78)

print("\n[1] INTÉRPRETE")
print(f"    sys.executable:       {sys.executable}")
print(f"    sys.prefix:           {sys.prefix}")
print(f"    sys.exec_prefix:      {sys.exec_prefix}")
print(f"    sys.base_prefix:      {sys.base_prefix}")
print(f"    sys.version:          {sys.version}")
print(f"    sys.version_info:     {sys.version_info}")
print(f"    platform.platform():  {platform.platform()}")
print(f"    sys.implementation:   {sys.implementation}")

print("\n[2] sys.path (en orden de búsqueda)")
for i, p in enumerate(sys.path):
    print(f"    [{i:02d}] {p!r}")
    print(f"          isdir={os.path.isdir(p) if p else False}")

print("\n[3] sys.meta_path (finders del import system)")
for f in sys.meta_path:
    print(f"    {f}")

print("\n[4] sys.path_hooks")
for f in sys.path_hooks:
    print(f"    {f}")

print("\n[5] sys.path_importer_cache (path -> importer)")
for k, v in sys.path_importer_cache.items():
    print(f"    {k!r} -> {v}")

print("\n[6] Sufijos que Python reconoce como módulo")
print(f"    EXTENSION_SUFFIXES:   {importlib.machinery.EXTENSION_SUFFIXES}")
print(f"    SOURCE_SUFFIXES:      {importlib.machinery.SOURCE_SUFFIXES}")
print(f"    BYTECODE_SUFFIXES:    {importlib.machinery.BYTECODE_SUFFIXES}")
print(f"    DEBUG_BYTECODE_SUFFIXES: {importlib.machinery.DEBUG_BYTECODE_SUFFIXES}")
print(f"    OPTIMIZED_BYTECODE_SUFFIXES: {importlib.machinery.OPTIMIZED_BYTECODE_SUFFIXES}")

print("\n[7] sysconfig (configuración de build de Python)")
for key in (
    "SOABI", "EXT_SUFFIX", "SHLIB_SUFFIX", "PYD_PLATFORM",
    "prefix", "exec_prefix", "BINDIR", "INCLUDEPY", "LIBDIR",
    "LDLIBRARY", "LIBRARY", "INSTSONAME",
    "CC", "CXX", "CFLAGS", "CCSHARED", "LDSHARED",
):
    try:
        v = sysconfig.get_config_var(key)
        print(f"    {key:20s} = {v!r}")
    except Exception as e:
        print(f"    {key:20s} = EXCEPTION: {e}")

print("\n[8] sysconfig.get_paths()")
for k, v in sysconfig.get_paths().items():
    print(f"    {k:20s} = {v!r}  isdir={os.path.isdir(v)}")

print("\n[9] site (rutas de site-packages)")
print(f"    site.ENABLE_USER_SITE: {site.ENABLE_USER_SITE}")
print(f"    site.USER_SITE:        {site.USER_SITE}")
print(f"    site.USER_BASE:        {site.USER_BASE}")
try:
    print("    site.getsitepackages():")
    for p in site.getsitepackages():
        print(f"        {p!r}  isdir={os.path.isdir(p)}")
except Exception as e:
    print(f"    site.getsitepackages() EXCEPTION: {e}")
try:
    print(f"    site.getusersitepackages(): {site.getusersitepackages()!r}")
except Exception as e:
    print(f"    site.getusersitepackages() EXCEPTION: {e}")

print("\n[10] Variables de entorno relevantes")
for k in (
    "PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP",
    "PYTHONVERBOSE", "PYTHONDEVMODE", "PYTHONNOUSERSITE",
    "PYTHONUSERBASE", "PYTHONSAFEPATH", "PYTHONUTF8",
    "PYTHONIOENCODING", "PYTHONCASEOK", "PYTHONHASHSEED",
    "CONDA_PREFIX", "CONDA_DEFAULT_ENV", "CONDA_SHLVL",
    "VIRTUAL_ENV", "LD_LIBRARY_PATH", "PATH",
):
    v = os.environ.get(k)
    print(f"    {k:20s} = {v!r}")

print("\n[11] sys.builtin_module_names (módulos compilados dentro del intérprete)")
names = sorted(sys.builtin_module_names)
print(f"    ({len(names)} total)")
for i in range(0, len(names), 8):
    print("    " + "  ".join(f"{n:20s}" for n in names[i:i+8]))

print("\n[12] sys.stdlib_module_names (todos los módulos de la stdlib)")
std = sorted(sys.stdlib_module_names)
print(f"    ({len(std)} total)")

print("\n[13] Módulos ya cargados (sys.modules)")
loaded = sorted(sys.modules.keys())
print(f"    ({len(loaded)} total)")
for i in range(0, len(loaded), 4):
    print("    " + "  ".join(f"{n:30s}" for n in loaded[i:i+4]))

print("\n[14] sys.flags")
for f in sys.flags:
    print(f"    {f}")

print("\n[15] sys.warnoptions")
for w in sys.warnoptions:
    print(f"    {w}")

print("\n[16] ¿Qué encuentra Python para nombres vacíos / especiales?")
for name in ("__main__", "site", "sys", "builtins", "importlib"):
    try:
        spec = importlib.util.find_spec(name)
        print(f"    find_spec({name!r}) -> {spec}")
    except Exception as e:
        print(f"    find_spec({name!r}) -> EXCEPTION {type(e).__name__}: {e}")

print("\n[17] Directorios de site-packages: contenido (primer nivel, .so y paquetes)")
for p in sys.path:
    if not p or not os.path.isdir(p):
        continue
    base = os.path.basename(p.rstrip("/"))
    if "site-packages" not in p and "dist-packages" not in p:
        continue
    print(f"\n    === {p} ===")
    try:
        entries = sorted(os.listdir(p))
    except Exception as e:
        print(f"        EXCEPTION: {e}")
        continue
    for e in entries:
        full = os.path.join(p, e)
        if e.endswith(".so") or ".so." in e or e.endswith(".pyd"):
            print(f"        [ext] {e}")
        elif os.path.isdir(full) and os.path.exists(os.path.join(full, "__init__.py")):
            print(f"        [pkg] {e}")

print("\n[18] .pth files en site-packages (modifican sys.path)")
for p in sys.path:
    if not p or not os.path.isdir(p):
        continue
    try:
        for f in os.listdir(p):
            if f.endswith(".pth"):
                full = os.path.join(p, f)
                print(f"\n    === {full} ===")
                try:
                    with open(full, "r", errors="replace") as fh:
                        for line in fh:
                            line = line.rstrip("\n")
                            if line and not line.startswith("#"):
                                print(f"        {line!r}  isdir={os.path.isdir(line)}")
                except Exception as e:
                    print(f"        EXCEPTION: {e}")
    except Exception:
        pass

print("\n[19] Verificación de que Python puede cargar C extensions")
print(f"    EXTENSION_SUFFIXES: {importlib.machinery.EXTENSION_SUFFIXES}")
print(f"    sysconfig SOABI:    {sysconfig.get_config_var('SOABI')}")
print(f"    sysconfig EXT_SUFFIX: {sysconfig.get_config_var('EXT_SUFFIX')}")

print("\n[20] Resumen de discrepancias sospechosas")
soabi = sysconfig.get_config_var("SOABI")
ext_suffix = sysconfig.get_config_var("EXT_SUFFIX")
print(f"    SOABI esperado:      {soabi}")
print(f"    EXT_SUFFIX esperado: {ext_suffix}")
print(f"    sys.executable:      {sys.executable}")
print(f"    sys.prefix:          {sys.prefix}")
if soabi and ext_suffix:
    print(f"    EXT_SUFFIX coincide con SOABI? {ext_suffix == '.' + soabi + '.so' if soabi else '?'}")

print("\n" + "=" * 78)
print("FIN")
print("=" * 78)