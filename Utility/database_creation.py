#######################################################################
##
## Script to create the training database for the neural network
## given synthetic images in linear_q and quadratic_q foldes where
## the images are stroed according to radiatl size in fodlers named
## a_x where x is the radius in meters. It assumes to have bakcground
## radiation data from Cherab in folder 'Cherab/hollow', with names
## a_x.jpg.
##
## Created by Soma Olasz, 05.2025
##
#######################################################################

import os
from PIL import Image
import numpy as np

import overlay_array as oa

# Input folders and model type labels
base_dirs = {
    "linear_q": "linear",
    "quadratic_q": "quadratic"
}

cherab_path = os.path.join("Cherab", "hollow")

# Ensure output directories exist
for model in base_dirs.values():
    os.makedirs(os.path.join("data_combined", "runaway", model), exist_ok=True)
os.makedirs(os.path.join("data_combined", "no_runaway"), exist_ok=True)

# Main processing loop
for input_dir, model_type in base_dirs.items():
    for a_folder in os.listdir(input_dir):
        a_path = os.path.join(input_dir, a_folder)
        if not os.path.isdir(a_path):
            continue

        png_path = os.path.join(cherab_path, f"{a_folder}.png")
        if not os.path.isfile(png_path):
            print(f"Missing Cherab image: {png_path}")
            continue

        cherab_img = Image.open(png_path)

        for subfolder in os.listdir(a_path):
            subfolder_path = os.path.join(a_path, subfolder)
            if not os.path.isdir(subfolder_path):
                continue

            jpeg_files = [f for f in os.listdir(subfolder_path) if f.endswith(".jpeg")]
            if not jpeg_files:
                continue

            jpeg_file = jpeg_files[0]
            jpeg_path = os.path.join(subfolder_path, jpeg_file)
            jpeg_img = Image.open(jpeg_path)

            for beta in np.arange(1, 7.5, 0.5):
                result_img, alpha = oa.combine_images(jpeg_img, cherab_img, beta, a_folder, add_noise=False)
                
                # Decide output path based on alpha
                if alpha >= 1.2:
                    out_folder = os.path.join("data_combined", "runaway", model_type)
                else:
                    out_folder = os.path.join("data_combined", "no_runaway")

                out_filename = f"{model_type}_{a_folder}_{subfolder}_{beta}.jpeg"
                out_path = os.path.join(out_folder, out_filename)

                result_img.save(out_path, "JPEG")

                
        print(f"{input_dir}_{a_folder} done")


print("Processing complete.")