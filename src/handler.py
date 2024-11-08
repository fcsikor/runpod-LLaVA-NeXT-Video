import json

import json_numpy
import runpod
from transformers import BitsAndBytesConfig
from transformers import LlavaNextVideoForConditionalGeneration
from transformers import LlavaNextVideoProcessor
import torch


quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

processor = LlavaNextVideoProcessor.from_pretrained(
    "llava-hf/LLaVA-NeXT-Video-7B-hf", cache_dir="/runpod-volume/")
model = LlavaNextVideoForConditionalGeneration.from_pretrained(
    "llava-hf/LLaVA-NeXT-Video-7B-hf",
    cache_dir="/runpod-volume/",
    quantization_config=quantization_config,
    device_map="auto"
)


def job_json2ndarray(job):
    video_ndarray = None

    if "multi_modal_data" in job:
        if "video" in job["multi_modal_data"]:
            video_str = json.dumps(
                job["multi_modal_data"]["video"])
            video_ndarray = json_numpy.loads(video_str)

    return video_ndarray


def handler(job):
    job_input = job["input"]

    if "sampling_params" in job_input:
        generate_kwargs = job_input["sampling_params"]
    else:
        generate_kwargs = {}

    clip = job_json2ndarray(job_input)

    if clip is None:  # no video input
        prompt = f"USER: {job_input['prompt']} ASSISTANT:"
        inputs = processor([prompt],
                           padding=True, return_tensors="pt").to(model.device)
    else:     # we received video input
        prompt = f"USER: <video>\n{job_input['prompt']} ASSISTANT:"
        inputs = processor([prompt], videos=[clip],
                           padding=True, return_tensors="pt").to(model.device)

    output = model.generate(**inputs, **generate_kwargs)
    generated_text = processor.batch_decode(output, skip_special_tokens=True)

    return generated_text


runpod.serverless.start({"handler": handler})
