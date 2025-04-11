# Marker enhancer
This repository contains code and trained models used for the paper ["Marker Data Enhancement for Markerless Motion Capture"](https://ieeexplore.ieee.org/document/10844513). 

The marker enhancer model predicts the 3D position of 43 anatomical markers (colored dots on right skeletons) from 20 video keypoint (colored triangles on left skeleton) positions. It consists of two models: the arm model predicts the 3D position of eight arm-located anatomical markers from seven arm and shoulder keypoint positions (blue) and the body model predicts the 3D position of 35 anatomical markers located on the shoulder, torso, and lower-body from 15 shoulder and lower-body keypoint positions (orange).

<p align="center">
  <img src="Images/MarkerEnhancer.png" width="800">
</p>

The marker enhancer is deployed as part of [OpenCap](https://www.opencap.ai/) ([source code](https://github.com/stanfordnmbl/opencap-core); [paper](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011462)), a markerless motion capture system to measure human movement from smartphone videos.

### Install requirements

1. Create environment: `conda create -n marker-enhancer python=3.11`
2. Activate environment: `conda activate marker-enhancer`
3. Install [TensorFlow with GPU support](https://www.tensorflow.org/install/pip).
    - For training/testing the Transformer model, you might need to use Ubuntu (at the time of doing this study, the required TensorFlow version was not supported on Windows)
4. Install other packages: `python -m pip install -r requirements.txt`

### Reference models
The paper compares three model architectures: LSTM, Transformer, and Linear.
The 3 resulting trained models are available in the `reference_models` folder.
The LSTM and Transformer performed best (more details in the paper). The LSTM
model is the default model used in OpenCap ([v0.3](https://github.com/stanfordnmbl/opencap-core/tree/main/MarkerAugmenter/LSTM))

### Data
The training data is split into separate datasets. Links to the public datasets are below.
- [Dataset0](https://drive.google.com/drive/folders/1ZpExo7fpdyX6qKjGvO40d8hJdm8906ij?usp=sharing) (corresponding to `Dataset0`, corresponding publication [here](https://pubmed.ncbi.nlm.nih.gov/27875132/))
- [Dataset1](https://drive.google.com/drive/folders/1_KLNAM5kO_UQXPBTRtHuvlH-eSyHXcHX?usp=sharing) (corresponding to `Dataset1`, corresponding publication [here](https://pubmed.ncbi.nlm.nih.gov/35798755/))
- [Dataset5](https://drive.google.com/drive/folders/1aQDwNCjPvcbaH9QWr-WkSotsYuBOww7T?usp=sharing) (corresponding to `Dataset5`, corresponding publication [here](https://royalsocietypublishing.org/doi/10.1098/rsif.2020.0487))
- [Dataset7](https://drive.google.com/drive/folders/1bNwpUqk_Igyc2l_iJwANraZeHS2ALnvT?usp=sharing) (corresponding to `Dataset7`, corresponding publication[here](https://pubmed.ncbi.nlm.nih.gov/31704899/))
- [Dataset8](https://drive.google.com/drive/folders/1bbse_zO24c-BOwZ9K9DoWlRVU8JXER57?usp=sharing) (corresponding to `Dataset8`, corresponding publication [here](https://pubmed.ncbi.nlm.nih.gov/33935233/))
- [Dataset9](https://drive.google.com/drive/folders/1ceqw0UWUrsLlQRT_HJHgJ0ANoicLwY_S?usp=sharing) (corresponding to `Dataset9`, corresponding publication [here](https://pubmed.ncbi.nlm.nih.gov/27793803/) and [here](https://pubmed.ncbi.nlm.nih.gov/29281799/))
- [Dataset10](https://drive.google.com/drive/folders/1dB1r5iqYTn7j7KGF-8w-KGjjokXgSUCQ?usp=sharing) (corresponding to `Dataset10`, corresponding dataset [here](http://mocap.cs.cmu.edu/))
- [Dataset11](https://drive.google.com/drive/folders/1dqIIZ_IkfJy-zBFdbXSTdljde6DU1szP?usp=sharing) (corresponding to `Dataset11`, corresponding publication [here](https://pubmed.ncbi.nlm.nih.gov/32948450/) and [here](https://pubmed.ncbi.nlm.nih.gov/33691592/))
- [Dataset23](https://drive.google.com/drive/folders/1kx_Nmtkb6L_ZXTYpKetD9hLgqgr0NgqA?usp=sharing) (corresponding to `Dataset23`, corresponding publication [here](https://pubmed.ncbi.nlm.nih.gov/35307884/))
- [Dataset24](https://drive.google.com/drive/folders/1ls5YUIZV9U5wlwMFku8EsHP4z-A2YBrY?usp=sharing) (corresponding to `Dataset24`, corresponding publication [here](https://pubmed.ncbi.nlm.nih.gov/31270327/))
- [Dataset25](https://drive.google.com/drive/folders/1mgfP8wdUJj9Z2Tjs2xd9UMCnhGi7fSm5?usp=sharing) (corresponding to `Dataset25`, corresponding publication [here](https://pmc.ncbi.nlm.nih.gov/articles/PMC6750598/))
- [Dataset26](https://drive.google.com/drive/folders/1naw5T5c-d4sG3wOLdBf_XTfqHR86Un4H?usp=sharing) (corresponding to `Dataset26`, corresponding publication [here](https://www.nature.com/articles/s41597-019-0323-z))
- [Dataset27](https://drive.google.com/drive/folders/1ob3MlNw9NjFxICHKk2lgCQw-s_RI-w6C?usp=sharing) (corresponding to `Dataset27`, corresponding publication [here](https://www.nature.com/articles/s41597-021-00801-5))

To use the datasets for training, download them and set them under the `Data/training_data_curated_60_openpose` folder.
You should have the following structure, for example:
- `Data/training_data_curated_60_openpose/h5_dataset0_60_openpose/infoData.npy`
- `Data/training_data_curated_60_openpose/h5_dataset0_60_openpose/time_sequences.h5`

### Training
The code we provide is for training the LSTM/Linear and Transformer models. 
1. LSTM or linear: `train_lstm_linear.py`
2. Transformer: `train_transformer.py`

### Testing
You can test the reference and trained models using the provided test script: `train_trained_models.py`
Follow the instructions at the top to test the reference models or your own trained models.

### Citation
If you use this code in your research, please cite our paper:
```bibtex
@ARTICLE{10844513,
  author={Falisse, Antoine and Uhlrich, Scott D. and Chaudhari, Akshay S. and Hicks, Jennifer L. and Delp, Scott L.},
  journal={IEEE Transactions on Biomedical Engineering},
  title={Marker Data Enhancement for Markerless Motion Capture},
  year={2025},
  volume={},
  number={},
  pages={1-10},
  keywords={Three-dimensional displays;Data models;Hip;Solid modeling;Accuracy;Training;Predictive models;Long short term memory;Computational modeling;Biomedical engineering;Deep learning;markerless motion capture;musculoskeletal modeling and simulation;pose estimation;trajectory optimization},
  doi={10.1109/TBME.2025.3530848}
}