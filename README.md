


# FoundationStereo使用
参考：https://github.com/NVlabs/FoundationStereo/tree/master?tab=readme-ov-file

conda env create -f environment.yml
conda run -n foundation_stereo pip install flash-attn
conda activate foundation_stereo


python scripts/run_demo.py --left_file ./assets/left.png --right_file ./assets/right.png --ckpt_dir ./pretrained_models/23-51-11/model_best_bp2.pth --out_dir ./test_outputs/



python scripts/run_demo.py --left_file ../imgs_left_rectify/WIN_20251207_14_54_12_Pro.png --right_file ../imgs_right_rectify/WIN_20251207_14_54_12_Pro.png --ckpt_dir ./pretrained_models/23-51-11/model_best_bp2.pth --out_dir ./test_outputs/
