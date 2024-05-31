echo "[Router] Selecting from the results..."
python ./router/router.py \
  --diffusion_results_folder="./results/diffusion" \
  --gan_results_folder="./results/gan" \
  --output_folder="./submission"
