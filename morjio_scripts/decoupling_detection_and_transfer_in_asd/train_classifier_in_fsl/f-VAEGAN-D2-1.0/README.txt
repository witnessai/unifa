#### f-VAEGAN-D2: A Feature Generating Framework for Any-Shot Learning. Yongqin Xian, Saurabh Sharma, Bernt Schiele, Zeynep Akata. IEEE CVPR 2019.

This repository includes the code (version 1.0) and data (CUB and FLO, finetuned ResNet101 features) for our CVPR 2019 paper on CNN visual features generation using f-VAEGAN-D2. Version 1.0 supports few-shot and generalized few-shot learning. We will complete the codes for zero-shot learning and generalized zero-shot learning soon. You can use it to reproduce the generalized few-shot learning results reported in the paper.

####How to reproduce the results:

1. Install Pytorch 0.4.0 and other dependencies 

2. Run one of the scripts, e.g. ./scripts/reproduce_vae_gan_d2_xu_fsl_cub.sh

If you want to finetune the ResNet yourself, please run ./scripts/finetune_cub.sh

####Citation

If you find this useful, please cite our work as follows:
@inproceedings {xianCVPR19a,     
 title = {f-VAEGAN-D2: A Feature Generating Framework for Any-Shot Learning},  
 booktitle = {IEEE Computer Vision and Pattern Recognition (CVPR)},     
 year = {2019},     
 author = {Yongqin Xian and Saurabh Sharma and Bernt Schiele and Zeynep Akata} 
} 

####Contact

Yongqin Xian
e-mail: yxian@mpi-inf.mpg.de
Computer Vision and Multimodal Computing, Max Planck Institute Informatics
Saarbruecken, Germany
https://xianyongqin.github.io/


####License

Copyright (c) 2019 Yongqin Xian, Max-Planck-Gesellschaft

Please read carefully the following terms and conditions and any accompanying documentation before you download and/or use this software and associated documentation files (the "Software").

The authors hereby grant you a non-exclusive, non-transferable, free of charge right to copy, modify, merge, publish, distribute, and sublicense the Software for the sole purpose of performing non-commercial scientific research, non-commercial education, or non-commercial artistic projects.

Any other use, in particular any use for commercial purposes, is prohibited. This includes, without limitation, incorporation in a commercial product, use in a commercial service, or production of other artefacts for commercial purposes. For commercial inquiries, please see above contact information.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

You understand and agree that the authors are under no obligation to provide either maintenance services, update services, notices of latent defects, or corrections of defects with regard to the Software. The authors nevertheless reserve the right to update, modify, or discontinue the Software at any time.

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. You agree to cite the f-VAEGAN-D2: A Feature Generating Framework for Any-Shot Learning paper in documents and papers that report on research using this Software.
