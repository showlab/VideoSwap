python propagate_point_displacement.py \
    --atlas_config_path="experiments/V6_final_evaluation/paper_results/object_atlas/6002_4_atlas_car_turn_inv_fp32/6002_4_atlas_car_turn_inv_fp32.yml" \
    --atlas_model_path="experiments/V6_final_evaluation/paper_results/object_atlas/6002_4_atlas_car_turn_inv_fp32/models/models_50000.pth" \
    --source_point_path="datasets/paper_evaluation/object/car_turn/annotation/00035.json" \
    --source_tap_path="datasets/paper_evaluation/object/car_turn/annotation/TAP.pth" \
    --target_point_path="datasets/paper_evaluation/object/car_turn/annotation/edit_point/00035_carA.json"
