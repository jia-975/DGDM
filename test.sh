#python gen_dg.py --score_config_path config/qm9_default.yml --discriminator_config_path config/qm9_dg_default.yml --w_dg 0.04
#
#python gen_dg.py --score_config_path config/qm9_default.yml --discriminator_config_path config/qm9_dg_default.yml --w_dg 0.06
#
#python gen_dg.py --score_config_path config/qm9_default.yml --discriminator_config_path config/qm9_dg_default.yml --w_dg 0.07

## qm9

#python gen_dg.py --score_config_path config/qm9_default.yml --discriminator_config_path config/qm9_dg_default.yml --w_dg 0.050 --seed 42
#python get_task1_results.py --input log/dg/qm9/ConfGF_epoch284min_sig0.000_dg_0.050_seed42.pkl --threshold 0.5
#python gen_dg.py --score_config_path config/qm9_default.yml --discriminator_config_path config/qm9_dg_default.yml --w_dg 0.050 --seed 729
#python get_task1_results.py --input log/dg/qm9/ConfGF_epoch284min_sig0.000_dg_0.050_seed729.pkl --threshold 0.5
# todo:记得更改yml文件中的checkpoints
python gen_dg.py --score_config_path config/qm9_default.yml --discriminator_config_path config/qm9_dg_default.yml --w_dg 0.050 --seed 2021
#python get_task1_results.py --input log/dg/qm9/ConfGF_epoch284min_sig0.000_dg_0.050_seed3407_reversenums800.pkl --threshold 0.5



#python gen_dg.py --score_config_path config/drugs_default.yml --discriminator_config_path config/drugs_dg_default.yml --w_dg 0.150 --seed 3407
#python get_task1_results.py --input log/dg/drugs/ConfGF_epoch284min_sig0.000_dg_0.150_seed3407.pkl --threshold 1.25

#python gen_dg.py --score_config_path config/qm9_default.yml --discriminator_config_path config/qm9_dg_default.yml --w_dg 0.050 --seed 729
#python get_task1_results.py --input log/dg/qm9/ConfGF_epoch284min_sig0.000_dg_0.050_seed729.pkl --threshold 0.5
#
#
#
#python gen_dg.py --score_config_path config/drugs_default.yml --discriminator_config_path config/drugs_dg_default.yml --w_dg 0.150 --seed 729
#python get_task1_results.py --input log/dg/drugs/ConfGF_epoch284min_sig0.000_dg_0.150_seed729.pkl --threshold 1.25











#python gen_dg.py --score_config_path config/qm9_default.yml --discriminator_config_path config/qm9_dg_default.yml --w_dg 0.030 --seed 2021
#python get_task1_results.py --input log/dg/qm9/ConfGF_epoch284min_sig0.000_dg_0.030_seed2021.pkl --threshold 0.5
#python gen_dg.py --score_config_path config/qm9_default.yml --discriminator_config_path config/qm9_dg_default.yml --w_dg 0.040 --seed 2021
#python get_task1_results.py --input log/dg/qm9/ConfGF_epoch284min_sig0.000_dg_0.040_seed2021.pkl --threshold 0.5
#python gen_dg.py --score_config_path config/qm9_default.yml --discriminator_config_path config/qm9_dg_default.yml --w_dg 0.060 --seed 2021
#python get_task1_results.py --input log/dg/qm9/ConfGF_epoch284min_sig0.000_dg_0.060_seed2021.pkl --threshold 0.5
#
#python gen_dg.py --score_config_path config/qm9_default.yml --discriminator_config_path config/qm9_dg_default.yml --w_dg 0.070 --seed 2021
#python get_task1_results.py --input log/dg/qm9/ConfGF_epoch284min_sig0.000_dg_0.070_seed2021.pkl --threshold 0.5
#
#python gen_dg.py --score_config_path config/qm9_default.yml --discriminator_config_path config/qm9_dg_default.yml --w_dg 0.090 --seed 2021
#python get_task1_results.py --input log/dg/qm9/ConfGF_epoch284min_sig0.000_dg_0.090_seed2021.pkl --threshold 0.5
#
#python gen_dg.py --score_config_path config/qm9_default.yml --discriminator_config_path config/qm9_dg_default.yml --w_dg 0.200 --seed 2021
#python get_task1_results.py --input log/dg/qm9/ConfGF_epoch284min_sig0.000_dg_0.200_seed2021.pkl --threshold 0.5
#
##python gen_dg.py --score_config_path config/qm9_default.yml --discriminator_config_path config/qm9_dg_default.yml --w_dg 0.08 --seed 2023
##python gen_dg.py --score_config_path config/qm9_default.yml --discriminator_config_path config/qm9_dg_default.yml --w_dg 0.08 --seed 2024
##python gen_dg.py --score_config_path config/qm9_default.yml --discriminator_config_path config/qm9_dg_default.yml --w_dg 0.08 --seed 2025
##
### drugs
#python gen_dg.py --score_config_path config/drugs_default.yml --discriminator_config_path config/drugs_dg_default.yml --w_dg 0.030 --seed 2021
#python get_task1_results.py --input log/dg/drugs/ConfGF_epoch284min_sig0.000_dg_0.030_seed2021.pkl --threshold 1.25
#python gen_dg.py --score_config_path config/drugs_default.yml --discriminator_config_path config/drugs_dg_default.yml --w_dg 0.040 --seed 2021
#python get_task1_results.py --input log/dg/drugs/ConfGF_epoch284min_sig0.000_dg_0.040_seed2021.pkl --threshold 1.25
#python gen_dg.py --score_config_path config/drugs_default.yml --discriminator_config_path config/drugs_dg_default.yml --w_dg 0.060 --seed 2021
#python get_task1_results.py --input log/dg/drugs/ConfGF_epoch284min_sig0.000_dg_0.060_seed2021.pkl --threshold 1.25
#python gen_dg.py --score_config_path config/drugs_default.yml --discriminator_config_path config/drugs_dg_default.yml --w_dg 0.070 --seed 2021
#python get_task1_results.py --input log/dg/drugs/ConfGF_epoch284min_sig0.000_dg_0.070_seed2021.pkl --threshold 1.25
#python gen_dg.py --score_config_path config/drugs_default.yml --discriminator_config_path config/drugs_dg_default.yml --w_dg 0.090 --seed 2021
#python get_task1_results.py --input log/dg/drugs/ConfGF_epoch284min_sig0.000_dg_0.090_seed2021.pkl --threshold 1.25
#python gen_dg.py --score_config_path config/drugs_default.yml --discriminator_config_path config/drugs_dg_default.yml --w_dg 0.200 --seed 2021
#python get_task1_results.py --input log/dg/drugs/ConfGF_epoch284min_sig0.000_dg_0.200_seed2021.pkl --threshold 1.25
#python gen_dg.py --score_config_path config/drugs_default.yml --discriminator_config_path config/drugs_dg_default.yml --w_dg 0.130 --seed 2021
#python get_task1_results.py --input log/dg/drugs/ConfGF_epoch284min_sig0.000_dg_0.130_seed2021.pkl --threshold 1.25


#python gen_dg.py --score_config_path config/drugs_default.yml --discriminator_config_path config/drugs_dg_default.yml --w_dg 0.15 --seed 2023
#python gen_dg.py --score_config_path config/drugs_default.yml --discriminator_config_path config/drugs_dg_default.yml --w_dg 0.15 --seed 2024
#python gen_dg.py --score_config_path config/drugs_default.yml --discriminator_config_path config/drugs_dg_default.yml --w_dg 0.15 --seed 2025
#
## evl
#python get_task1_results.py --input log/dg/qm9/ConfGF_epoch284min_sig0.000_dg_0.08_seed2022.pkl
#python get_task1_results.py --input log/dg/qm9/ConfGF_epoch284min_sig0.000_dg_0.08_seed2023.pkl
#python get_task1_results.py --input log/dg/qm9/ConfGF_epoch284min_sig0.000_dg_0.08_seed2024.pkl
#python get_task1_results.py --input log/dg/qm9/ConfGF_epoch284min_sig0.000_dg_0.08_seed2025.pkl

#python get_task1_results.py --input log/dg/drugs/ConfGF_epoch284min_sig0.000_dg_0.15_seed2022.pkl --threshold 1.25
#python get_task1_results.py --input log/dg/drugs/ConfGF_epoch284min_sig0.000_dg_0.15_seed2023.pkl --threshold 1.25
#python get_task1_results.py --input log/dg/drugs/ConfGF_epoch284min_sig0.000_dg_0.15_seed2024.pkl --threshold 1.25
#python get_task1_results.py --input log/dg/drugs/ConfGF_epoch284min_sig0.000_dg_0.15_seed2025.pkl --threshold 1.25
#
#python get_task1_results.py --input log/dg/qm9/ConfGF_qm9_epoch284min_sig0.000_dg_0.06.txt
#
#python get_task1_results.py --input log/dg/qm9/ConfGF_qm9_epoch284min_sig0.000_dg_0.07.txt
#
#python get_task1_results.py --input log/dg/qm9/ConfGF_qm9_epoch284min_sig0.000_dg_0.09.txt




#python -u gen.py --config_path ./config/qm9_default.yml --generator ConfGF --start 0 --end 200 --num_repeat 50 --test_set qm9_property.pkl
#
#python gen_dg.py --score_config_path config/qm9_default.yml --discriminator_config_path config/qm9_dg_default.yml --w_dg 0.05 --seed 2021 --start 0 --end 200 --num_repeat 50 --test_set qm9_property.pkl
