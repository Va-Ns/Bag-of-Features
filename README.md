# Bag of Features Image Classification in MATLAB

This project is inspired by the code snippets available from the [ICCV of 2005](https://people.csail.mit.edu/fergus/iccv2005/bagwords.html). It implements a Bag of Features model for image classification in MATLAB. The model involves several steps including feature extraction, vector quantization, and classification using the intrinsic functions of MATLAB. 

## Project Highlights

- **Feature Extraction**: Utilizes SIFT features to identify points of interest in images.
- **Vector Quantization**: Employs k-means clustering to quantize feature vectors.
- **Classification**: Uses a Support Vector Machine classifier with a grid search for the optimization of its parameters.

## Project Structure

- `Bag_Of_Features.m`: The main script that orchestrates the model execution.
- `BoW_final.mlx`: A MATLAB Live Script with detailed explanations and results.
- `compute_descriptors.ln`, `discrete_sampler.m`, `vgg_*`: Various utility scripts for feature extraction and processing.
- `Edge_Sampling.m`, `Edge_Sampling_Vasilakis.m`: Scripts for edge sampling techniques.
- `images/`: Directory containing image datasets for training and testing the model.

## How to Run

To run this project:
1. Ensure MATLAB is installed on your system.
2. Clone this repository to your local machine.
3. Open MATLAB and navigate to the cloned project directory.
4. Run the `Bag_Of_Features.m` script to start the image classification model.

## Disclaimer

This repository is a simple form of reproduction with quite a few changes compared to the initial files that are provided in the ICCV. In keeping with that theme, one can identify files, such as `compute_descriptors.ln`, `discrete_sampler.m`, `vgg_*`, that belong in the initial draft of the contributors yet are important for the completion of this work. All acknowledgements for those files go to the authors! 

## License

Following the notion of the authors, this code is for teaching/research purposes only.

<img src = "https://github.com/NickTy-byte/Bag-of-Features/assets/68824495/9650581c-220e-4629-a115-e3b0cc97cda7" width="500" height="500"> 
<img src = "https://github.com/NickTy-byte/Bag-of-Features/assets/68824495/5bfa844f-2bf3-4dd8-8340-688351ea1d6c" width="500" height="500"> 
<img src = "https://github.com/NickTy-byte/Bag-of-Features/assets/68824495/a2118543-6eb1-4280-84be-6417bb587c22" width="1000" height="1000"> 

![image](https://github.com/NickTy-byte/Bag-of-Features/assets/68824495/5bfa844f-2bf3-4dd8-8340-688351ea1d6c)
![image](https://github.com/NickTy-byte/Bag-of-Features/assets/68824495/a2118543-6eb1-4280-84be-6417bb587c22)


