from pathlib import Path
import argparse
import soundfile as sf
import torch
import io
import argparse
from matcha.hifigan.config import v1
from matcha.hifigan.denoiser import Denoiser
from matcha.hifigan.env import AttrDict
from matcha.hifigan.models import Generator as HiFiGAN
from matcha.models.matcha_tts import MatchaTTS
from matcha.text import sequence_to_text, text_to_sequence
from matcha.utils.utils import intersperse
import gradio as gr




def load_matcha( checkpoint_path, device):
    model = MatchaTTS.load_from_checkpoint(checkpoint_path, map_location=device)
    _ = model.eval()
    return model

def load_hifigan(checkpoint_path, device):
    h = AttrDict(v1)
    hifigan = HiFiGAN(h).to(device)
    hifigan.load_state_dict(torch.load(checkpoint_path, map_location=device)["generator"])
    _ = hifigan.eval()
    hifigan.remove_weight_norm()
    return hifigan

def load_vocoder(checkpoint_path, device):
    vocoder = None
    vocoder = load_hifigan(checkpoint_path, device)
    denoiser = Denoiser(vocoder, mode="zeros")
    return vocoder, denoiser

def process_text(i: int, text: str, device: torch.device):
    print(f"[{i}] - Input text: {text}")
    x = torch.tensor(
        intersperse(text_to_sequence(text, ["kyrgyz_cleaners"]), 0),
        dtype=torch.long,
        device=device,
    )[None]
    x_lengths = torch.tensor([x.shape[-1]], dtype=torch.long, device=device)
    x_phones = sequence_to_text(x.squeeze(0).tolist())
    print(f"[{i}] - Phonetised text: {x_phones[1::2]}")
    return {"x_orig": text, "x": x, "x_lengths": x_lengths, "x_phones": x_phones}

def to_waveform(mel, vocoder, denoiser=None):
    audio = vocoder(mel).clamp(-1, 1)
    if denoiser is not None:
        audio = denoiser(audio.squeeze(), strength=0.00025).cpu().squeeze()
    return audio.cpu().squeeze()

@torch.inference_mode()
def process_text_gradio(text):
    output = process_text(1, text, device)
    return output["x_phones"][1::2], output["x"], output["x_lengths"]

@torch.inference_mode()
def synthesise_mel(text, text_length, n_timesteps, temperature, length_scale, spk=-1):
    spk = torch.tensor([spk], device=device, dtype=torch.long) if spk >= 0 else None
    output = model.synthesise(
        text,
        text_length,
        n_timesteps=n_timesteps,
        temperature=temperature,
        spks=spk,
        length_scale=length_scale,
    )
    output["waveform"] = to_waveform(output["mel"], vocoder, denoiser)
    return output["waveform"].numpy()

def get_inference(text, n_timesteps=20, mel_temp = 0.667, length_scale=0.8, spk=-1):
    phones, text, text_lengths = process_text_gradio(text)
    print(type(synthesise_mel(text, text_lengths, n_timesteps, mel_temp, length_scale, spk)))
    return synthesise_mel(text, text_lengths, n_timesteps, mel_temp, length_scale, spk)


device = torch.device("cpu")
model_path = './checkpoints/checkpoint.ckpt'
vocoder_path = './checkpoints/generator'
model = load_matcha(model_path, device) 
vocoder, denoiser = load_vocoder(vocoder_path, device) 

def gen_tts(text, speaking_rate):
    return 22050, get_inference(text = text, length_scale = speaking_rate)

default_text = "Баарыңарга салам, менин атым Акылай."

css = """
        #share-btn-container {
            display: flex;
            padding-left: 0.5rem !important;
            padding-right: 0.5rem !important;
            background-color: #000000;
            justify-content: center;
            align-items: center;
            border-radius: 9999px !important; 
            width: 13rem;
            margin-top: 10px;
            margin-left: auto;
            flex: unset !important;
        }
        #share-btn {
            all: initial;
            color: #ffffff;
            font-weight: 600;
            cursor: pointer;
            font-family: 'IBM Plex Sans', sans-serif;
            margin-left: 0.5rem !important;
            padding-top: 0.25rem !important;
            padding-bottom: 0.25rem !important;
            right:0;
        }
        #share-btn * {
            all: unset !important;
        }
        #share-btn-container div:nth-child(-n+2){
            width: auto !important;
            min-height: 0px !important;
        }
        #share-btn-container .wrap {
            display: none !important;
        }
"""
with gr.Blocks(css=css) as block:
    gr.HTML(
        """
            <div style="text-align: center; max-width: 700px; margin: 0 auto;">
              <div
                style="
                  display: inline-flex; align-items: center; gap: 0.8rem; font-size: 1.75rem;
                "
              >
                <h1 style="font-weight: 900; margin-bottom: 7px; line-height: normal;">
                  Akyl-AI TTS
                </h1>
              </div>
            </div>
        """
    )
    with gr.Row():
        image_url = "https://github.com/simonlobgromov/Matcha-TTS/blob/main/photo_2024-04-07_15-59-52.png?raw=true"
        gr.Image(image_url, label=None, width=660, height=315, show_label=False)
    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(label="Input Text", lines=2, value=default_text, elem_id="input_text")
            speaking_rate = gr.Slider(label='Speaking rate', minimum=0.5, maximum=1, step=0.05, value=0.8, interactive=True, show_label=True, elem_id="speaking_rate")


            run_button = gr.Button("Generate Audio", variant="primary")
        with gr.Column():
            audio_out = gr.Audio(label="Parler-TTS generation", type="numpy", elem_id="audio_out")

    inputs = [input_text, speaking_rate]
    outputs = [audio_out]
    run_button.click(fn=gen_tts, inputs=inputs, outputs=outputs, queue=True)


block.queue()
block.launch(share=True)
