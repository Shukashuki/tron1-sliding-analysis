from pathlib import Path
import json,hashlib,csv,io,sys
ROOT=Path(__file__).resolve().parents[2]
PATHS={'i_ppo': 'results/ground_contact_v2i_ppo_nominal_20261024_v1/analysis.json', 'i_cem': 'results/ground_contact_v2i_cem_nominal_20261024_v1/cem_analysis.json', 'j_ppo': 'results/ground_contact_v2j_ppo_nominal_20261024_v1/analysis.json', 'j_cem': 'results/ground_contact_v2j_cem_nominal_20261024_v2/cem_analysis.json', 'k_ppo': 'results/ground_contact_v2k_ppo_nominal_20261024_v1/analysis.json', 'k_cem': 'results/ground_contact_v2k_cem_nominal_20261024_v1/cem_analysis.json'}
PINS={'i_ppo': 'c72d227b9a4e12d5faebba079b684c96ff201a29c4cde0573ace515f46cfce75', 'i_cem': '9730d7f3a29d1c863a8fea9271318ec3f54d0ff07410954a9d3da54c6da65c79', 'j_ppo': '45fe45720d1ff112485a701175f25c6f6b5ad877fd796d21f99ec478d381cbc5', 'j_cem': '30414df33c9c44a557dc82db3d5cdabcf6b960f4858f35c02362e05ba62b23d9', 'k_ppo': 'c01cff93c49229b418b8a3e13944ae5fadc81c8a22a5df3fa2ff53aa6467d928', 'k_cem': '1c0bd5e9d330a5d39a0eaafd62014499e86a3d9d7f68ab2aa3d93d921b270146'}
COLUMNS=['cohort', 'controller', 'engine', 'core_xy_rmse_m', 'whole_xy_rmse_m', 'whole_xy_peak_m', 'core_speed_ratio', 'core_beta_rmse_deg', 'full_anywheel_support_fraction', 'full_unsupported_samples', 'full_max_gap_s', 'core_HF_ratio_to_original_same_controller_engine', 'failed_checks']
def comparison_rows(data):
 rows=[]
 for label in ['i','j','k']:
  for kind in ['ppo','cem']:
   summary=data[label+'_'+kind]
   items=summary['rows'] if kind=='ppo' else [summary]
   for item in items:
    value=item['new_V2'] if kind=='ppo' else item['actual_V2'+label]
    core=value['windows']['core']['tracking'];whole=value['windows']['whole_reference_overlap']['tracking']
    contact=value['any_wheel_ground_support']['full_contact'];screen=item['nominal_screen']
    rows.append(dict(zip(COLUMNS,[label.upper(),kind.upper(),item['validation_backend'] if kind=='ppo' else 'native CPU MuJoCo',
     core['xy_rmse_m'],whole['xy_rmse_m'],whole['xy_peak_m'],core['speed_retention'],core['beta_rmse_deg'],
     contact['any_wheel_support_sample_fraction'],contact['unsupported_physical_samples'],
     contact['longest_no_support_observed_s'],screen['core_HF_ratio_to_original_same_engine_PPO' if kind=='ppo' else 'core_HF_ratio_to_old_CEM'],
     '|'.join(screen['failed_checks'])])))
 return rows

def main():
 data={}
 for key,path in PATHS.items():
  raw=(ROOT/path).read_bytes()
  assert hashlib.sha256(raw).hexdigest()==PINS[key], 'Pinned public analysis changed: '+path
  data[key]=json.loads(raw)
 buffer=io.StringIO();writer=csv.DictWriter(buffer,fieldnames=COLUMNS,lineterminator='\n')
 writer.writeheader();writer.writerows(comparison_rows(data));sys.stdout.write(buffer.getvalue())
if __name__=='__main__':main()
