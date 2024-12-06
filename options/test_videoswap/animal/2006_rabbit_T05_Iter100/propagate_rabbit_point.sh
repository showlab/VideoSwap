#python propagate_point_displacement.py \
#    --atlas_config_path="experiments/V6_final_evaluation/paper_results/animal_atlas/4022_4_atlas_rabbit_inv_fp32/4022_4_atlas_rabbit_inv_fp32.yml" \
#    --atlas_model_path="experiments/V6_final_evaluation/paper_results/animal_atlas/4022_4_atlas_rabbit_inv_fp32/models/models_50000.pth" \
#    --source_point_path="datasets/paper_evaluation/animal/rabbit/annotation/00034.json" \
#    --source_tap_path="datasets/paper_evaluation/animal/rabbit/annotation/TAP.pth" \
#    --target_point_path="datasets/paper_evaluation/animal/rabbit/annotation/edit_point/00034_dogB.json"

python propagate_point_displacement.py \
    --atlas_config_path="experiments/V6_final_evaluation/paper_results/animal_atlas/4022_4_atlas_rabbit_inv_fp32/4022_4_atlas_rabbit_inv_fp32.yml" \
    --atlas_model_path="experiments/V6_final_evaluation/paper_results/animal_atlas/4022_4_atlas_rabbit_inv_fp32/models/models_50000.pth" \
    --source_point_path="datasets/paper_evaluation/animal/rabbit/annotation/00034.json" \
    --source_tap_path="datasets/paper_evaluation/animal/rabbit/annotation/TAP.pth" \
    --target_point_path="datasets/paper_evaluation/animal/rabbit/annotation/edit_point/00034_catA.json"
