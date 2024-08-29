# Overview and Details of Datasets

Following, we provide a more detailed overview on the dataset collection of `Robust-Minisets`. For further details and statistics on the novel datasets Tiny ImageNet-A, -R and -v2 please refer to our [paper](https://arxiv.org/abs/2408.14131).

### CIFAR-10.1
`Description:` CIFAR-10.1 (v4) is a new test set for the CIFAR-10 dataset introduced by Recht et al. ("Do CIFAR-10 Classifiers Generalize to CIFAR-10?") with 2,000 images spanning all 10 classes of the original CIFAR-10 dataset. CIFAR-10.1 consists of images sampled from the TinyImages dataset based on keyword search and count. For further information visit the original GitHub repository of [CIFAR-10.1](https://github.com/modestyachts/CIFAR-10.1).

`Number of Classes:` 10

`Resolution:` 32×32

`Splits:` Test (2,021)

`License:` MIT License

### CIFAR-10-C
`Description:` CIFAR-10-C is an open-source data set comprising algorithmically generated corruptions applied to the CIFAR-10 test set comprising 10 classes following the concept of ImageNet-C. It was introduced by Hendrycks et al. ("Benchmarking Neural Network Robustness to Common Corruptions and Perturbations") and comprises 19 different corruptions (15 test corruptions and 4 validation corruptions) spanning 5 severity levels. This results in 200,000 images for the validation set and 750,000 images for the test set. For further information visit the original GitHub repository of [ImageNet-C](https://github.com/hendrycks/robustness).

`Number of Classes:` 10

`Resolution:` 32×32

`Splits:` Validation (200,000) / Test (750,000)

`Corruptions:`
* Validation: ['speckle_noise', 'gaussian_blur', 'spatter', 'saturate']
* Test: ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']

`License:` Creative Commons Attribution 4.0 International

### CIFAR-100-C
`Description:` CIFAR-100-C is an open-source data set comprising algorithmically generated corruptions applied to the CIFAR-100 test set comprising 100 classes following the concept of ImageNet-C. It was introduced by Hendrycks et al. ("Benchmarking Neural Network Robustness to Common Corruptions and Perturbations") and comprises 19 different corruptions (15 test corruptions and 4 validation corruptions) spanning 5 severity levels. This results in 200,000 images for the validation set and 750,000 images for the test set. For further information visit the original GitHub repository of [ImageNet-C](https://github.com/hendrycks/robustness).

`Number of Classes:` 100

`Resolution:` 32×32

`Splits:` Validation (200,000) / Test (750,000)

`Corruptions:`
* Validation: ['speckle_noise', 'gaussian_blur', 'spatter', 'saturate']
* Test: ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']

`License:` Creative Commons Attribution 4.0 International

### EuroSAT
`Description:` EuroSAT is a dataset and deep learning benchmark for land use and land cover classification. The dataset is based on Sentinel-2 satellite images consisting of 10 classes with in total 27,000 labeled samples. For the train/val/test split we follow Neumann et al. (["In-Domain Representation Learning for Remote Sensing"](https://github.com/google-research/google-research/tree/master/remote_sensing_representations), Google Research). For further information visit the original GitHub repository of [EuroSAT](https://github.com/phelber/EuroSAT).

`Number of Classes:` 10

`Resolution:` 32×32

`Splits:` Train (16,200) / Validation (5,400) / Test (5,400)

`License:` MIT License

### EuroSAT-C
`Description:` EuroSAT-C is an open-source data set comprising algorithmically generated corruptions applied to the EuroSAT test set following the concept of ImageNet-C. It comprises 19 different corruptions (15 test corruptions and 4 validation corruptions) spanning 5 severity levels resulting in 108,000 images for the validation set and 405,000 images for the test set. For further information on the corruptions visit the original GitHub repository of [ImageNet-C](https://github.com/hendrycks/robustness).

`Number of Classes:` 10

`Resolution:` 64×64

`Splits:` Validation (108,000) / Test (405,000)

`Corruptions:`
* Validation: ['speckle_noise', 'gaussian_blur', 'spatter', 'saturate']
* Test: ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']

`License:` MIT License

### BloodMNIST-C
`Description:` BloodMNIST-C is an open-source data set comprising algorithmically generated corruptions applied to the BloodMNIST test set of the [MedMNIST](https://medmnist.com) collection following the concept of ImageNet-C. It comprises 16 different corruptions (12 test corruptions and 4 validation corruptions) spanning 5 severity levels resulting in 68,420 images for the validation set and 205,260 images for the test set. For further information on the corruptions visit the original GitHub repository of [ImageNet-C](https://github.com/hendrycks/robustness).

`Number of Classes:` 8

`Resolution:` 28×28

`Splits:` Validation (68,420) / Test (205,260)

`Corruptions:`
* Validation: ['speckle_noise', 'gaussian_blur', 'spatter', 'saturate']
* Test: ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']

`License:` Creative Commons Attribution 4.0 International

### BreastMNIST-C
`Description:` BreastMNIST-C is an open-source data set comprising algorithmically generated corruptions applied to the BreastMNIST test set of the [MedMNIST](https://medmnist.com) collection following the concept of ImageNet-C. It comprises 16 different corruptions (12 test corruptions and 4 validation corruptions) spanning 5 severity levels resulting in 3,120 images for the validation set and 9,360 images for the test set. For further information on the corruptions visit the original GitHub repository of [ImageNet-C](https://github.com/hendrycks/robustness).

`Number of Classes:` 2

`Resolution:` 28×28

`Splits:` Validation (3,120) / Test (9,360)

`Corruptions:`
* Validation: ['speckle_noise', 'gaussian_blur', 'spatter', 'saturate']
* Test: ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']

`License:` Creative Commons Attribution 4.0 International

### DermaMNIST-C
`Description:` DermaMNIST-C is an open-source data set comprising algorithmically generated corruptions applied to the DermaMNIST test set of the [MedMNIST](https://medmnist.com) collection following the concept of ImageNet-C. It comprises 16 different corruptions (12 test corruptions and 4 validation corruptions) spanning 5 severity levels resulting in 40,100 images for the validation set and 120,300 images for the test set. For further information on the corruptions visit the original GitHub repository of [ImageNet-C](https://github.com/hendrycks/robustness).

`Number of Classes:` 7

`Resolution:` 28×28

`Splits:` Validation (40,100) / Test (120,300)

`Corruptions:`
* Validation: ['speckle_noise', 'gaussian_blur', 'spatter', 'saturate']
* Test: ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']

`License:` Creative Commons Attribution-NonCommercial 4.0 International

### OCTMNIST-C
`Description:` OCTMNIST-C is an open-source data set comprising algorithmically generated corruptions applied to the OCTMNIST test set of the [MedMNIST](https://medmnist.com) collection following the concept of ImageNet-C. It comprises 16 different corruptions (12 test corruptions and 4 validation corruptions) spanning 5 severity levels resulting in 20,000 images for the validation set and 60,000 images for the test set. For further information on the corruptions visit the original GitHub repository of [ImageNet-C](https://github.com/hendrycks/robustness).

`Number of Classes:` 4

`Resolution:` 28×28

`Splits:` Validation (20,000) / Test (60,000)

`Corruptions:`
* Validation: ['speckle_noise', 'gaussian_blur', 'spatter', 'saturate']
* Test: ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']

`License:` Creative Commons Attribution 4.0 International

### OrganAMNIST-C
`Description:` OrganAMNIST-C is an open-source data set comprising algorithmically generated corruptions applied to the OrganAMNIST test set of the [MedMNIST](https://medmnist.com) collection following the concept of ImageNet-C. It comprises 16 different corruptions (12 test corruptions and 4 validation corruptions) spanning 5 severity levels resulting in 355,560 images for the validation set and 1,066,680 images for the test set. For further information on the corruptions visit the original GitHub repository of [ImageNet-C](https://github.com/hendrycks/robustness).

`Number of Classes:` 11

`Resolution:` 28×28

`Splits:` Validation (355,560) / Test (1,066,680)

`Corruptions:`
* Validation: ['speckle_noise', 'gaussian_blur', 'spatter', 'saturate']
* Test: ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']

`License:` Creative Commons Attribution 4.0 International

### OrganCMNIST-C
`Description:` OrganCMNIST-C is an open-source data set comprising algorithmically generated corruptions applied to the OrganCMNIST test set of the [MedMNIST](https://medmnist.com) collection following the concept of ImageNet-C. It comprises 16 different corruptions (12 test corruptions and 4 validation corruptions) spanning 5 severity levels resulting in 164,320 images for the validation set and 492,960 images for the test set. For further information on the corruptions visit the original GitHub repository of [ImageNet-C](https://github.com/hendrycks/robustness).

`Number of Classes:` 11

`Resolution:` 28×28

`Splits:` Validation (164,320) / Test (492,960)

`Corruptions:`
* Validation: ['speckle_noise', 'gaussian_blur', 'spatter', 'saturate']
* Test: ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']

`License:` Creative Commons Attribution 4.0 International

### OrganSMNIST-C
`Description:` OrganSMNIST-C is an open-source data set comprising algorithmically generated corruptions applied to the OrganSMNIST test set of the [MedMNIST](https://medmnist.com) collection following the concept of ImageNet-C. It comprises 16 different corruptions (12 test corruptions and 4 validation corruptions) spanning 5 severity levels resulting in 176,540 images for the validation set and 529,620 images for the test set. For further information on the corruptions visit the original GitHub repository of [ImageNet-C](https://github.com/hendrycks/robustness).

`Number of Classes:` 11

`Resolution:` 28×28

`Splits:` Validation (176,540) / Test (529,620)

`Corruptions:`
* Validation: ['speckle_noise', 'gaussian_blur', 'spatter', 'saturate']
* Test: ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']

`License:` Creative Commons Attribution 4.0 International

### PathMNIST-C
`Description:` PathMNIST-C is an open-source data set comprising algorithmically generated corruptions applied to the PathMNIST test set of the [MedMNIST](https://medmnist.com) collection following the concept of ImageNet-C. It comprises 16 different corruptions (12 test corruptions and 4 validation corruptions) spanning 5 severity levels resulting in 143,600 images for the validation set and 430,800 images for the test set. For further information on the corruptions visit the original GitHub repository of [ImageNet-C](https://github.com/hendrycks/robustness).

`Number of Classes:` 9

`Resolution:` 28×28

`Splits:` Validation (143,600) / Test (430,800)

`Corruptions:`
* Validation: ['speckle_noise', 'gaussian_blur', 'spatter', 'saturate']
* Test: ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']

`License:` Creative Commons Attribution 4.0 International

### PneumoniaMNIST-C
`Description:` PneumoniaMNIST-C is an open-source data set comprising algorithmically generated corruptions applied to the PneumoniaMNIST test set of the [MedMNIST](https://medmnist.com) collection following the concept of ImageNet-C. It comprises 16 different corruptions (12 test corruptions and 4 validation corruptions) spanning 5 severity levels resulting in 12,480 images for the validation set and 37,440 images for the test set. For further information on the corruptions visit the original GitHub repository of [ImageNet-C](https://github.com/hendrycks/robustness).

`Number of Classes:` 2

`Resolution:` 28×28

`Splits:` Validation (12,480) / Test (37,440)

`Corruptions:`
* Validation: ['speckle_noise', 'gaussian_blur', 'spatter', 'saturate']
* Test: ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']

`License:` Creative Commons Attribution 4.0 International

### TissueMNIST-C
`Description:` TissueMNIST-C is an open-source data set comprising algorithmically generated corruptions applied to the TissueMNIST test set of the [MedMNIST](https://medmnist.com) collection following the concept of ImageNet-C. It comprises 16 different corruptions (12 test corruptions and 4 validation corruptions) spanning 5 severity levels resulting in 945,600 images for the validation set and 2,836,800 images for the test set. For further information on the corruptions visit the original GitHub repository of [ImageNet-C](https://github.com/hendrycks/robustness).

`Number of Classes:` 8

`Resolution:` 28×28

`Splits:` Validation (68,420) / Test (205,260)

`Corruptions:`
* Validation: ['speckle_noise', 'gaussian_blur', 'spatter', 'saturate']
* Test: ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']

`License:` Creative Commons Attribution 4.0 International

### Tiny ImageNet
`Description:` Tiny Imagenet is a scaled down version of the ImageNet dataset and was created by folks at the Stanford University. Tiny ImageNet contains 100,000 images of 200 classes (500 for each class) downsized to 64×64 colored images. Each class has 500 training images and 50 test images.

`Number of Classes:` 200

`Resolution:` 64×64

`Splits:` Train (100,000) / Test (10,000)

`License:`

### Tiny ImageNet-A
`Description:` Tiny ImageNet-A is a subset of the Tiny ImageNet test set consisting of 3,374 images comprising real-world, unmodified, and naturally occurring examples that are misclassified by ResNet-18. The sampling process of Tiny ImageNet-A roughly follows the concept of ImageNet-A introduced by Hendrycks et al. ("Natural Adversarial Examples"). For further information visit the original GitHub repository of [ImageNet-A](https://github.com/hendrycks/natural-adv-examples).

`Number of Classes:` 200

`Resolution:` 64×64

`Splits:` Test (3,374)

`License:`

### Tiny ImageNet-C
`Description:` Tiny ImageNet-C is an open-source data set comprising algorithmically generated corruptions applied to the Tiny ImageNet test set comprising 200 classes following the concept of ImageNet-C. It was introduced by Hendrycks et al. ("Benchmarking Neural Network Robustness to Common Corruptions and Perturbations") and comprises 19 different corruptions (15 test corruptions and 4 validation corruptions) spanning 5 severity levels. This results in 200,000 images for the validation set and 750,000 images for the test set. For further information visit the original GitHub repository of [ImageNet-C](https://github.com/hendrycks/robustness).

`Number of Classes:` 200

`Resolution:` 64×64

`Splits:` Validation (200,000) / Test (750,000)

`Corruptions:`
* Validation: ['speckle_noise', 'gaussian_blur', 'spatter', 'saturate']
* Test: ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']

`License:` Creative Commons Attribution 4.0 International

### Tiny ImageNet-R
`Description:` Tiny ImageNet-R is a subset of the ImageNet-R dataset by Hendrycks et al. ("The Many Faces of Robustness: A Critical Analysis of Out-of-Distribution Generalization") with 10,456 images spanning 62 of the 200 Tiny ImageNet dataset. It is a test set achieved by collecting images of joint classes of Tiny ImageNet and ImageNet. The resized images of size 64×64 contain art, cartoons, deviantart, graffiti, embroidery, graphics, origami, paintings, patterns, plastic objects, plush objects, sculptures, sketches, tattoos, toys, and video game renditions of ImageNet classes. For further information visit the original GitHub repository of [ImageNet-R](https://github.com/hendrycks/imagenet-r).

`Number of Classes:` 200

`Resolution:` 64×64

`Splits:` Test (10,456)

`License:` MIT License

### Tiny ImageNetv2
`Description:` Tiny ImageNetv2 is a subset of the ImageNetV2 (matched frequency) dataset by Recht et al. ("Do ImageNet Classifiers Generalize to ImageNet?") with 2,000 images spanning all 200 classes of the Tiny ImageNet dataset. It is a test set achieved by collecting images of joint classes of Tiny ImageNet and ImageNet. The resized images of size 64×64 consist of images collected from Flickr after a decade of progress on the original ImageNet dataset. The data collection process was designed to resemble the original ImageNet dataset distribution. For further information visit the original GitHub repository of [ImageNetV2](https://github.com/modestyachts/ImageNetV2).

`Number of Classes:` 200

`Resolution:` 64×64

`Splits:` Test (2,000)

`License:` MIT License