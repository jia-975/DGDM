python gen.py --config_path ./config/qm9_default.yml --generator ConfGF  --start 0 --end 200


python gen_dg.py --score_config_path config/drugs_default.yml --discriminator_config_path config/drugs_dg_default.yml --w_dg 0.01





python gen_dg.py --score_config_path config/qm9_default.yml --discriminator_config_path config/qm9_dg_default.yml --w_dg 0.1

python gen.py --config_path ./config/qm9_default.yml --generator ConfGF  --start 0 --end 200
