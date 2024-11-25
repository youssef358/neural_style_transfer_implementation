# Neural Style Transfer Implementation

## 1. Implementation of Neural Style Transfer

This project implements the Image Style Transfer Using Convolutional Neural Networks algorithm proposed in the research paper by **Gatys et al.**, leveraging deep convolutional neural networks to blend the content of one image with the style of another. The method optimizes an output image to simultaneously match the content representation of a "content image" and the style representation of a "style image" as captured by a pre-trained network.

---

## 2. General Overview of the Project

This implementation is structured into four main classes to modularize functionality and facilitate maintainability:

- **ImageProcessor**: Handles image preprocessing (content, style, and output images) and postprocessing tasks such as loading, resizing, and saving images.
- **LossFunctions**: Contains functions for calculating the content loss, style loss, and total variation loss required to optimize the generated image.
- **Optimizer**: Prepares and manages the L-BFGS optimizer, defining the optimization steps required to minimize the defined loss function.
- **NeuralStyleTransfer**: The core class that integrates all components and contains the `run` method to perform the style transfer.

---

## 3. Project Structure

The project is organized as follows:



```
neural_style_transfer_implementation/
├───data
│   ├───content
    ├───output
│   ├───style
├───src
├───neural_style_transfer.ipynb  # Jupyter Notebook for running experiments
```

---

## 4. Results

### Example 1: Generated Images from Content and Style Pairings

Below is an example showcasing the output of the algorithm. The content image is blended with five different style images mentioned in the **Gatys et al.** paper.

| ![result_3](data/output/nst_Tuebingen_der_schrei/Tuebingen_der_schrei_h_400_cw_1800_sw_2.0_tv_0.0.jpg)| ![result_1](data/output/nst_Tuebingen_the_shipwreck_of_the_minotaur/Tuebingen_the_shipwreck_of_the_minotaur_h_400_cw_1000000.0_sw_5000.0_tv_1.0.jpg) | ![result_2](data/output/nst_Tuebingen_Starry_Night/Tuebingen_Starry_Night_h_400_cw_1000000.0_sw_1000.0_tv_1.0.jpg) |
|---------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
|![result_4](data/output/nst_Tuebingen_figure_dans_un_fauteuil/Tuebingen_figure_dans_un_fauteuil_h_400_cw_1000000.0_sw_4000.0_tv_1.0.jpg)             | ![result_5](data/output/nst_Tuebingen_Composition_7/Tuebingen_Composition_7_h_400_cw_5000000.0_sw_4000.0_tv_1.0.jpg)

### Example 2: Alternate Content + Style + Output Visualization

| **Content** | **Style** | **Generated Output** |
|-------------|-----------|----------------------|
| ![content](data/content/nature_1.jpg) | ![style](data/style/the_embarkation_for_cythera.jpg) | ![output](data/output/nst_nature_1_the_embarkation_for_cythera/nature_1_the_embarkation_for_cythera_h_400_cw_1000000.0_sw_1000.0_tv_1.0.jpg) |

### Video Illustration

For a dynamic visualization of the style transfer process, refer to the included GIF:
<p align="center">
  <img src="data/output/nst_nature_1_the_embarkation_for_cythera/nature_1_the_embarkation_for_cythera_h_400_cw_1000000.0_sw_1000.0_tv_1.0.gif" alt="Demo">
</p>

---

## References


- Gatys, L.A., Ecker, A.S., & Bethge, M. *Image Style Transfer Using Convolutional Neural Networks.* Available [Here](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Gatys_Image_Style_Transfer_CVPR_2016_paper.pdf).
