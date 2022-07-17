https://www.kaggle.com/competitions/gan-getting-started/overview

# I'm Something of a Painter Myself - 
### Orel Ben-Zaken and Omer Lux

We recognize the works of artists through their unique style, such as color choices or brush strokes. The “je ne sais quoi” of artists like Claude Monet can now be imitated with algorithms thanks to generative adversarial networks (GANs). In this getting started competition, you will bring that style to your photos or recreate the style from scratch!

Computer vision has advanced tremendously in recent years and GANs are now capable of mimicking objects in a very convincing way. But creating museum-worthy masterpieces is thought of to be, well, more art than science. So can (data) science, in the form of GANs, trick classifiers into believing you’ve created a true Monet? That’s the challenge you’ll take on!


## The Challenge:

A GAN consists of at least two neural networks: a generator model and a discriminator model. The generator is a neural network that creates the images. For our competition, you should generate images in the style of Monet. This generator is trained using a discriminator.

The two models will work against each other, with the generator trying to trick the discriminator, and the discriminator trying to accurately classify the real vs. generated images.

Your task is to build a GAN that generates 7,000 to 10,000 Monet-style images.


## Implementations:

We used the Cycle-GAN [[2](https://openaccess.thecvf.com/content_ICCV_2017/papers/Zhu_Unpaired_Image-To-Image_Translation_ICCV_2017_paper.pdf)], [[3](https://openaccess.thecvf.com/content_cvpr_2017/papers/Isola_Image-To-Image_Translation_With_CVPR_2017_paper.pdf)] architecture in order to address this problem. More information on the [report](./report.pdf).
![Cycle-GAN Example Architecture](./pic/cyclegan.png)

### Improvements that did better:

- [x] Dataset Augmentations (pre-feedforward).
- [x] Differentiable Augmentation between Generators and Discriminators [[4](https://proceedings.neurips.cc/paper/2020/file/55479c55ebd1efd3ff125f1337100388-Paper.pdf)].
- [x] Noise addition for self-adversarial attacks [[5](https://proceedings.neurips.cc/paper/2019/file/b83aac23b9528732c23cc7352950e880-Paper.pdf)].
- [x] Dual discriminators for GANs [[6](https://proceedings.neurips.cc/paper/2017/file/e60e81c4cbe5171cd654662d9887aec2-Paper.pdf)]

## Acknowledgement

1. [Kaggle Monet Cyclegan Tutorial Notebook](https://proceedings.neurips.cc/paper/2020/file/55479c55ebd1efd3ff125f1337100388-Paper.pdf)
2. [Zhu, Jun-Yan, et al. "Unpaired image-to-image translation using cycle-consistent adversarial networks." Proceedings of the IEEE international conference on computer vision. 2017](https://openaccess.thecvf.com/content_ICCV_2017/papers/Zhu_Unpaired_Image-To-Image_Translation_ICCV_2017_paper.pdf)
3. [Isola, Phillip, et al. "Image-to-image translation with conditional adversarial networks." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017](https://openaccess.thecvf.com/content_cvpr_2017/papers/Isola_Image-To-Image_Translation_With_CVPR_2017_paper.pdf)
4. [Zhao, Shengyu, et al. "Differentiable augmentation for data-efficient gan training." Advances in Neural Information Processing Systems 33 (2020)](https://proceedings.neurips.cc/paper/2020/file/55479c55ebd1efd3ff125f1337100388-Paper.pdf)
5. [Bashkirova, Dina, Ben Usman, and Kate Saenko. "Adversarial self-defense for cycle-consistent GANs." Advances in Neural Information Processing Systems 32 (2019)](https://proceedings.neurips.cc/paper/2019/file/b83aac23b9528732c23cc7352950e880-Paper.pdf)
6. [Nguyen, Tu, et al. "Dual discriminator generative adversarial nets." Advances in neural information processing systems 30 (2017)](https://proceedings.neurips.cc/paper/2017/file/e60e81c4cbe5171cd654662d9887aec2-Paper.pdf)