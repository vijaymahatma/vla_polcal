#Re-apply calibration tables, but now setting parang=False
import os
import subprocess
import logging
import sys
#import astropy
import matplotlib.pyplot as plt
import ast
import csv
import run_casatasks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("polcal")

def load_calibration_config(config_path):
    config = {}
    with open(config_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue  # Skip empty lines and comments
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            try:
                config[key] = ast.literal_eval(value)
            except Exception as e:
                raise ValueError(f"Failed to parse line: {line}\n{e}")
    return config

if len(sys.argv) < 6:
    print("Usage: python vla_polcal.py <msin> <msout> <msout_target> <output_dir> <config_file.py>")
    sys.exit(1)

msin = sys.argv[1]
msout = sys.argv[2]
msout_target = sys.argv[3]
output_dir = sys.argv[4]
config_file = sys.argv[5]

#msin="24B-425.sb47044525.eb47312503.60639.24434767361.ms"
#msout='24B-425_C_calib_polcal.ms'
#msout_target='24B-425_C_calib_polcal_3C20.ms'

config = load_calibration_config(config_file)

band = config.get("band")
all_spws = config.get("all_spws")
band_spws = config.get("band_spws")
final_spws = config.get("final_spws")
sources = config.get("sources")
cal_leakage = config.get("cal_leakage")
cal_leakage_im = config.get("cal_leakage_im")
cal_leakage_newgains = config.get("cal_leakage_newgains")
cal_leakage_I_model = config.get("cal_leakage_I_model")
cal_leakage_I_model_spix = config.get("cal_leakage_I_model_spix")
cal_leakage_ref_freq = config.get("cal_leakage_ref_freq")
cal_polangle = config.get("cal_polangle")
cal_polangle_ref_freq = config.get("cal_polangle_ref_freq")
target = config.get("target")
skip_initcal = config.get('skip_initcal')
refant = config.get('refant')

if band=="Ku":
	baseband_list=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32,32]
	solve_bbc1 = '0~15:5~58'
	solve_bbc2 = '16~31:5~58'
	solve_bbc3 = '32~47:5~58'
	band_ref = 15.0
elif band=="C":
	baseband_list=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,16,16,16,16,16,16,16,16,16,16,26,26,26,26,26,26]
	solve_bbc1 = '0~15:5~58'
	solve_bbc2 = '16~25:5~58'
	solve_bbc3 = '26~31:5~58'
	band_ref = 6.0
elif band=="L":
	baseband_list=[0,0,0,0,0,0,0,0,8,8,8,8,8,8,8,8]
	solve_bbc1 = "0~7:5~58"
	solve_bbc2 = "8~15:5~58"
	solve_bbc3 = None
	band_ref = 1.5

casa_path = "/soft/casa-latest/bin/casa"

def run_command(cmd, shell=False):
	try:
		logger.info(f"Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
		subprocess.run(cmd, shell=shell, check=True)
	except subprocess.CalledProcessError as e:
		logger.error(f"Command failed: {e}")
		raise

# Function to run CASA commands
def run_casa_command(casa_script):
	casa_cmd = f"{casa_path} --nogui -c {casa_script}"
	run_command(casa_cmd, shell=True)

#os.system(casa_path)
os.chdir(output_dir)

casa_script = run_casatasks.generate_casa_script(
    msin, msout, msout_target,
    all_spws, band_spws, final_spws,
    sources, refant, band,
    cal_leakage, cal_leakage_im, cal_leakage_newgains,
    cal_leakage_I_model, cal_leakage_I_model_spix, cal_leakage_ref_freq,
    cal_polangle, cal_polangle_ref_freq,
    skip_initcal, baseband_list,
    solve_bbc1, solve_bbc2, solve_bbc3, band_ref,target
)

script_path = f"{output_dir}/casa_polcal.py"
with open(script_path, "w") as f:
	f.write(casa_script)

# Run CASA script for calibration
run_casa_command(script_path)


"""
script_path = f"{output_dir}/casa_polcal.py"
with open(script_path, "w") as f:
	f.write(casa_script)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("polcal")
# Run CASA script for calibration
run_casa_command(script_path)

logger.info("Pol cal completed successfully!")
"""