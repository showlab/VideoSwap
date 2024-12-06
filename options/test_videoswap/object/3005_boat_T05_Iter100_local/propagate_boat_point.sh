python propagate_point_displacement.py \
    --atlas_config_path="experiments/V6_final_evaluation/paper_results/object_atlas/6031_4_atlas_boat_inv_fp32/6031_4_atlas_boat_inv_fp32.yml" \
    --atlas_model_path="experiments/V6_final_evaluation/paper_results/object_atlas/6031_4_atlas_boat_inv_fp32/models/models_50000.pth" \
    --source_point_path="datasets/paper_evaluation/object/boat/annotation/00000.json" \
    --source_tap_path="datasets/paper_evaluation/object/boat/annotation/TAP.pth" \
    --target_point_path="datasets/paper_evaluation/object/boat/annotation/edit_point/00000_sailboat2.json"

#python propagate_point_displacement.py \
#    --atlas_config_path="experiments/V6_final_evaluation/paper_results/object_atlas/6031_4_atlas_boat_inv_fp32/6031_4_atlas_boat_inv_fp32.yml" \
#    --atlas_model_path="experiments/V6_final_evaluation/paper_results/object_atlas/6031_4_atlas_boat_inv_fp32/models/models_50000.pth" \
#    --source_point_path="datasets/paper_evaluation/object/boat/annotation/00000.json" \
#    --source_tap_path="datasets/paper_evaluation/object/boat/annotation/TAP.pth" \
#    --target_point_path="datasets/paper_evaluation/object/boat/annotation/edit_point/00000_yacht.json"
