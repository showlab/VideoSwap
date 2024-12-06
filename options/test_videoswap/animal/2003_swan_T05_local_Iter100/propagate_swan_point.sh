python propagate_point_displacement.py \
    --atlas_config_path="experiments/V6_final_evaluation/paper_results/animal_atlas/4032_4_atlas_swan_inv_fp32/4032_4_atlas_swan_inv_fp32.yml" \
    --atlas_model_path="experiments/V6_final_evaluation/paper_results/animal_atlas/4032_4_atlas_swan_inv_fp32/models/models_40000.pth" \
    --source_point_path="datasets/paper_evaluation/animal/blackswan/annotation/00000.json" \
    --source_tap_path="datasets/paper_evaluation/animal/blackswan/annotation/TAP.pth" \
    --target_point_path="datasets/paper_evaluation/animal/blackswan/annotation/edit_point/00000_catA.json"

python propagate_point_displacement.py \
    --atlas_config_path="experiments/V6_final_evaluation/paper_results/animal_atlas/4032_4_atlas_swan_inv_fp32/4032_4_atlas_swan_inv_fp32.yml" \
    --atlas_model_path="experiments/V6_final_evaluation/paper_results/animal_atlas/4032_4_atlas_swan_inv_fp32/models/models_40000.pth" \
    --source_point_path="datasets/paper_evaluation/animal/blackswan/annotation/00000.json" \
    --source_tap_path="datasets/paper_evaluation/animal/blackswan/annotation/TAP.pth" \
    --target_point_path="datasets/paper_evaluation/animal/blackswan/annotation/edit_point/00000_duck.json"

python propagate_point_displacement.py \
    --atlas_config_path="experiments/V6_final_evaluation/paper_results/animal_atlas/4032_4_atlas_swan_inv_fp32/4032_4_atlas_swan_inv_fp32.yml" \
    --atlas_model_path="experiments/V6_final_evaluation/paper_results/animal_atlas/4032_4_atlas_swan_inv_fp32/models/models_40000.pth" \
    --source_point_path="datasets/paper_evaluation/animal/blackswan/annotation/00000.json" \
    --source_tap_path="datasets/paper_evaluation/animal/blackswan/annotation/TAP.pth" \
    --target_point_path="datasets/paper_evaluation/animal/blackswan/annotation/edit_point/00000_dogA.json"
