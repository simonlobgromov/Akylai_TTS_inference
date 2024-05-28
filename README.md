<div align="center">



# AkylAI TTS


[![python](https://img.shields.io/badge/-Python_3.10-blue?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3100/)
[![pytorch](https://img.shields.io/badge/PyTorch_2.0+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![lightning](https://img.shields.io/badge/-Lightning_2.0+-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![hydra](https://img.shields.io/badge/Config-Hydra_1.3-89b8cd)](https://hydra.cc/)
[![black](https://img.shields.io/badge/Code%20Style-Black-black.svg?labelColor=gray)](https://black.readthedocs.io/en/stable/)
[![isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)

<p style="text-align: center;">
  <img src="https://github.com/simonlobgromov/Matcha-TTS/blob/main/photo_2024-04-07_15-59-52.png" height="400"/>
</p>

</div>



# AkylAI-TTS for Kyrgyz language

We present to you a model trained in the Kyrgyz language, which has been trained on 13 hours of speech and 7,000 samples, complete with source code and training scripts. The architecture is based on Matcha-TTS.
It`s a new approach to non-autoregressive neural TTS, that uses [conditional flow matching](https://arxiv.org/abs/2210.02747) (similar to [rectified flows](https://arxiv.org/abs/2209.03003)) to speed up ODE-based speech synthesis. Our method:

- Is probabilistic
- Has compact memory footprint
- Sounds highly natural
- Is very fast to synthesise from

You can try our *AkylAI TTS* by visiting [SPACE](https://huggingface.co/spaces/the-cramer-project/akylai-tts-mini) and read [ICASSP 2024 paper](https://arxiv.org/abs/2309.03199) for more details.

# Inference

It is recommended to start by setting up a virtual environment using `venv`.

1. Clone this repository and install all modules and dependencies by running the commands:

```bash
git clone https://github.com/simonlobgromov/Akylai_inference
cd Akylai_inference
pip install -e .
apt-get install espeak-ng
```
2. Load checkpoints

```bash
mkdir checkpoints
cd checkpoints
```

for example
```
wget "https://github.com/simonlobgromov/AkylAI_Matcha_Checkpoint/releases/download/Matcha-TTS/checkpoint_epoch.499.ckpt" -O "checkpoint.ckpt"
wget "https://github.com/simonlobgromov/AkylAI_Matcha_HiFiGan/releases/download/Generator/generator_v1" -O "generator"
```

then

```bash
cd ..
mkdir output
```

3. Run inference

```bash
python inference_script.py --text '<your text>'
```
`inference_script.py` help:

```
options:
  -h, --help            show this help message and exit
  --text TEXT           Text to synthesize
  --speaking_rate SPEAKING_RATE
                        change the speaking rate, a higher value means slower speaking rate
                        (default: 0.8)
```

The audio will be saved to the `./Akylai_inference/output` folder!!

The `inference_script.py` also has a `tensor_to_wav_bytes()` function that converts output to a byte file.









# Credits


- Shivam Mehta ([GitHub](https://github.com/shivammehta25))
- The Cramer Project (Data collection and preprocessing)[Official Space](https://thecramer.com/)
- Amantur Amatov (Expert)
- Timur Turatali (Expert, Research)
- Den Pavlov (Research, Data preprocessing and ML engineering) [GitHub](https://github.com/simonlobgromov/Matcha-TTS))
- Ulan Abdurazakov (Environment Developer)
- Nursultan Bakashov (CEO)

## Citation information

If you use our code or otherwise find this work useful, please cite our paper:

```text
@inproceedings{mehta2024matcha,
  title={Matcha-{TTS}: A fast {TTS} architecture with conditional flow matching},
  author={Mehta, Shivam and Tu, Ruibo and Beskow, Jonas and Sz{\'e}kely, {\'E}va and Henter, Gustav Eje},
  booktitle={Proc. ICASSP},
  year={2024}
}
```

## Acknowledgements

Since this code uses [Lightning-Hydra-Template](https://github.com/ashleve/lightning-hydra-template), you have all the powers that come with it.

Other source code we would like to acknowledge:

- [Coqui-TTS](https://github.com/coqui-ai/TTS/tree/dev): For helping me figure out how to make cython binaries pip installable and encouragement
- [Hugging Face Diffusers](https://huggingface.co/): For their awesome diffusers library and its components
- [Grad-TTS](https://github.com/huawei-noah/Speech-Backbones/tree/main/Grad-TTS): For the monotonic alignment search source code
- [torchdyn](https://github.com/DiffEqML/torchdyn): Useful for trying other ODE solvers during research and development
- [labml.ai](https://nn.labml.ai/transformers/rope/index.html): For the RoPE implementation

