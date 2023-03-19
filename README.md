# European to Asian CycleGAN

![Cover image](https://i.imgur.com/4yzDJau.png)

## Introduction

CycleGAN is a neural network architecture that enables transformations
from one image domain to another without requiring paired images for
training. In this project, the aim is to use CycleGAN to transform 
the appearance of European faces into Asian faces while retaining 
the original identity.

## Implementation

The project is implemented in Python 3 using the PyTorch deep 
learning framework. Real-time tracking of the learning process
is implemented using TensorBoard. The original CycleGAN 
architecture was used, with minor modifications to the generator
architecture to optimize calculations.

The project architecture consists of the following files:

- `config.py`: Configuration file where hyperparameters, training settings, augmentation, and other settings can be changed.
- `train.py`: Main training cycle of the neural network, including loading and saving checkpoints during training.
- `generator.py`: Contains the generator class.
- `discriminator.py`: Contains the discriminator class.
- `dataset.py`: Implements the dataset class.
- `test_dataset.py`: Tests the dataset and augmentations.
- `utils.py`: Contains auxiliary functions such as checkpoint loading and image post-processing.
- `download_files.py`: Script that downloads the dataset, pre-trained weights, and test images in one click.

## Dataset

The dataset used in this project is a custom dataset consisting 
of 16,384 images of faces, with 8,192 stereotypically European 
faces and 8,192 stereotypically Asian faces. The dataset includes
both male and female individuals, with a main age range of 20-40
years.

#### Class A: 8192 European<br> 
   - ~4096 Female (~10-70)
   - ~4096 Male (~9-70)

![Картинка][european_dataset]


#### Class B: 8192 Asian<br> 
   - ~4096 Female (~16-70)
   - ~4096 Male (~10-65)

![Картинка][asian_dataset]


## Results

The model successfully transforms the appearance of European 
faces into Asian faces while retaining the original identity. 
The resulting images have characteristic Asian features while 
maintaining the basic features of the original European faces.

![Results of the model inference](https://i.imgur.com/y3xpAnZ.png)

## Training Tips

Some tips for training the model include:

- Training the model for 30 epochs with standard config settings and then reducing the learning rate to zero for 10 epochs.
- Adjusting the number of residual layers when resizing the input image (e.g., using six residual layers for 128x128 images).
- Leaving the batch size parameter equal to one as increasing the batch size does not significantly increase the convergence rate of the model.
- Training the model using GPU.

## Colab Notebook

A Colab notebook is available to quickly test the pre-trained model inference:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1tgz_iiSEL-iSf1DCM4lXCJ0WJUT061FS?usp=sharing)

## Ethical Considerations

It is important to consider the ethical implications of any project 
involving sensitive topics such as racial and cultural identity. 
While this project focuses on transforming the appearance of European
faces into Asian faces, it is important to acknowledge that there 
is no single "Asian appearance" and that individuals within the same
cultural and racial group can have a wide range of physical appearances.
It is crucial to approach such topics with sensitivity and to avoid 
reinforcing harmful stereotypes or promoting discriminatory attitudes.

The dataset used in this project was filtered for stereotypical 
features, but it is important to be mindful of how the resulting 
images are interpreted and used. Furthermore, this project is purely
for academic and research purposes, and it is not intended to be 
used for any harmful or discriminatory actions.


## References

- Zhu, J., Park, T., Isola, P., & Efros, A. A. (2017). Unpaired image-to-image translation using cycle-consistent adversarial networks. In Proceedings of the IEEE international conference on computer vision (pp. 2223-2232).

## Contact Information

If you have any questions or feedback, please feel free to contact me:

[![Mail](https://i.imgur.com/HILZFT2.png)](mailto:tutikgv@gmail.com)
**E-mail:**
[tutikgv@gmail.com](mailto:tutikgv@gmail.com) <br>

[![Telegram](https://i.imgur.com/IMICyTA.png)](https://t.me/glebtutik)
**Telegram:**
https://t.me/glebtutik <br>


[european_dataset]: https://i.imgur.com/ZFswgK9.png "Датасет european"
[asian_dataset]: https://i.imgur.com/6tG6lEg.png "Датасет asian"
