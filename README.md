# Human Face Sketch to Real Image Translation using Conditional Generative Adversarial Networks
The Pix2pix CGAN model proposed in the paper (https://arxiv.org/abs/1611.07004) [Image-to-Image Translation with Conditional Adversarial Networks] by Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, Alexei A. Efros has been implemented in this project.
The Discriminator uses PatchGAN and the Generator uses a U-Net architecture.

The dataset used is the [CUHK-Face Dataset] (mmlab.ie.cuhk.edu.hk/archive/cufsf/) consisting of 188 face-sketch pairs.

The zip files with data ready to be fed for preprocessing can be found in **asianstrain_256.npz** for training data and **asianstest_256.npz** for test data.

The **cgan_code2_train_on_152_imgs.ipynb** notebook consists of the preprocessing and full architecture code.

The saved trained model **g_model_045600.h5** can be used to generate samples for test dataset using **cgan_on_testset.ipynb** notebook.
Similarly the same model can be used to generate custom results for custom input using **cgan_on_custom_input.ipynb**.

After cloning the repository, a simple flask app to test the model on custom input can be initialized by `flask run` inside the current repo from the terminal.


