python -u train.py -s data/ref_nerf/ref_synthetic/ball --eval --iterations 61000 --white_background
python -u train.py -s data/ref_nerf/ref_synthetic/car --eval --iterations 61000 --white_background
python -u train.py -s data/ref_nerf/ref_synthetic/coffee --eval --iterations 61000 --white_background
python -u train.py -s data/ref_nerf/ref_synthetic/helmet --eval --iterations 61000 --white_background
python -u train.py -s data/ref_nerf/ref_synthetic/teapot --eval --iterations 61000 --white_background
python -u train.py -s data/ref_nerf/ref_synthetic/toaster --eval --iterations 61000 --white_background --longer_prop_iter 24_000

python -u train.py -s data/nero/GlossySynthetic/angel_blender --eval --iterations 61000 --white_background --longer_prop_iter 36_000
python -u train.py -s data/nero/GlossySynthetic/bell_blender --eval --iterations 91000 --white_background --longer_prop_iter 48_000  --opac_lr0_interval 0
python -u train.py -s data/nero/GlossySynthetic/cat_blender --eval --iterations 61000 --white_background
python -u train.py -s data/nero/GlossySynthetic/horse_blender --eval --iterations 61000 --white_background --longer_prop_iter 36_000
python -u train.py -s data/nero/GlossySynthetic/luyu_blender --eval --iterations 61000 --white_background
python -u train.py -s data/nero/GlossySynthetic/potion_blender --eval --iterations 61000 --white_background --longer_prop_iter 24_000 
python -u train.py -s data/nero/GlossySynthetic/tbell_blender --eval --iterations 61000 --white_background  --longer_prop_iter 36_000  --opac_lr0_interval 0
python -u train.py -s data/nero/GlossySynthetic/teapot_blender --eval --iterations 61000 --white_background --longer_prop_iter 36_000

python -u train.py -s data/ref_nerf/ref_real/gardenspheres --eval --iterations 61000  --longer_prop_iter 36_000 --use_env_scope --env_scope_center -0.2270 1.9700 1.7740 --env_scope_radius 0.974
python -u train.py -s data/ref_nerf/ref_real/sedan --eval --iterations 61000 --longer_prop_iter 36_000 --use_env_scope --env_scope_center -0.032 0.808 0.751 --env_scope_radius 2.138
python -u train.py -s data/ref_nerf/ref_real/toycar --eval --iterations 61000  --longer_prop_iter 36_000 --use_env_scope --env_scope_center 0.6810 0.8080 4.4550 --env_scope_radius 2.707


python -u train.p-s data/nerf_synthetic/lego --eval --iterations 61000 --white_background --densification_interval_when_prop 100
python -u train.py -s data/nerf_synthetic/drums --eval --iterations 61000 --white_background --densification_interval_when_prop 100
python -u train.py -s data/nerf_synthetic/ship --eval --iterations 61000 --white_background --densification_interval_when_prop 100
python -u train.py -s data/nerf_synthetic/hotdog --eval --iterations 61000 --white_background --densification_interval_when_prop 100
python -u train.py -s data/nerf_synthetic/ficus --eval --iterations 61000 --white_background --densification_interval_when_prop 100
python -u train.py -s data/nerf_synthetic/mic --eval --iterations 61000 --white_background --densification_interval_when_prop 100