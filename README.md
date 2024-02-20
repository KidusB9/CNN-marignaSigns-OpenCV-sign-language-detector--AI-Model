# Sign Language Recognition Using OpenCV and CNN

This project aims to recognize sign language gestures using Convolutional Neural Networks (CNN) and OpenCV, making communication more accessible for the deaf and hard-of-hearing community.

## Dependencies

To run this project, you need to have the following libraries installed:

- `TensorFlow`
- `Keras`
- `OpenCV`

or just run pip install requirments.txt

## Dataset

The dataset used for training the model is the Sign Language MNIST available on Kaggle. It consists of images of hand gestures representing letters of the English alphabet.

[Sign Language MNIST Dataset](https://www.kaggle.com/datamunge/sign-language-mnist)

## Tools

For this project, [Google Colab](https://colab.research.google.com/) is recommended due to its free access to GPUs, making the training process faster.

## How to Run

1. **ROIinOpenCV.py**: This script prepares the Region of Interest (ROI) using OpenCV for real-time gesture recognition. Run this script to start capturing hand gestures.

    ```bash
    python ROIinOpenCV.py
    ```

2. **PyTorch Implementation**: For those interested in a PyTorch version, `sign_language_pytorch.ipynb` is provided. This Jupyter Notebook contains a step-by-step guide to implementing the CNN model using Py


    To run the notebook:

    - Upload `sign_language_pytorch.ipynb` to Google Colab.
    - Ensure the dataset is accessible to the Colab notebook, either by uploading it directly or mounting your Google Drive.
    - Follow the instructions within the notebook to train and evaluate the model.

## Project Structure

- `ROIinOpenCV.py`: Script for real-time gesture recognition using OpenCV.
- `sign_language_pytorch.ipynb`: Jupyter Notebook for training and evaluating the CNN model using PyTorch.

## Installation

To set up your environment to run the code, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/Kidus-berhanu/CNNY-marignaSigns-OpenCV-sign-language-detector--AI-mode.git
    ```

2. Install the required Python packages:
    ```bash
    pip install tensorflow keras opencv-python
    ```

3. Run the `ROIinOpenCV.py` script or the Jupyter Notebook as per your preference.

## Contributing

Contributions to improve the project are welcome. Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes and commit them (`git commit -am 'Add some feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.

## Acknowledgments

- Special thanks to the creators of the Sign Language MNIST dataset for providing a great resource for sign language recognition research.
- Gratitude to Google Colab for offering a platform with free GPU access, facilitating deep learning experiments.

---
