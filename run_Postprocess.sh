echo "Postprocessing results..."
python postprocess.py \
  --diffusion_results_folder="./results/diffusion/HR" \
  --gan_results_folder="./results/gan" \
  --output_folder="./submission"
