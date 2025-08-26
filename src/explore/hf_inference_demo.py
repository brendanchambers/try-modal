
import modal

# configure base image.  
# takes ~ few minutes to build unless cached
cuda_version = "12.8.1"  # should be no greater than host CUDA version
flavor = "devel"  # includes full CUDA toolkit
operating_sys = "ubuntu24.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"
HF_CACHE_PATH = "/cache"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.12")
    .pip_install(
        "vllm==0.9.1",
        "huggingface_hub[hf_transfer]==0.32.0",
        "flashinfer-python==0.2.6.post1",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "0",  # 1 -> true
          "HF_HUB_CACHE": '/cache/hf/hub',
          "HF_HOME": '/cache/hf/home',
        #   "TORCH_CUDA_ARCH_LIST": "9.0 9.0a",  # H100, silence noisy logs
          })  # faster model transfers
)
app = modal.App(image=image)



@app.function(gpu="T4:1")
def healthcheck():
    import torch
    assert torch.cuda.is_available()
    import subprocess
    output = subprocess.check_output(["nvidia-smi"], text=True)
    assert "Driver Version:" in output
    assert "CUDA Version:" in output
    print(output)
    return output


######## remote: excerise_hf_model_generate
@app.function(gpu='T4:1')
def exercise_hf_model_generate():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
    print('finished loading tokenizer')
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B-Base",  torch_dtype=torch.float16)
                                                 #, cache_dir='/cache/hf/hub/models')
    print('finished loading model')
    messages = [
        {"role": "user", "content": "Unenlightened one: 'Please tell us something inscrutable, wise one.'\nSage: 'My siblings, you have traveled far. However... Instead of"}, # trailing space is very bad, avoid trailing space
    ]
    print('tokenizing...')
    # you can use a chat template if you like
    # inputs = tokenizer.apply_chat_template(
    #     messages,
    #     add_generation_prompt=False,
    #     tokenize=True,
    #     return_dict=True,
    #     return_tensors="pt",
    # ).to(model.device)
    # tokenize only
    inputs = tokenizer(
        messages[0]['content'],
        return_tensors="pt",
    ).to(model.device)

    print('generating...')
    outputs = model.generate(**inputs, max_new_tokens=100)
    print('decoding generation...')
    print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))
    print('ehfmg')

    print('round trip tokenization:')
    print(inputs['input_ids'])
    print(inputs['input_ids'].cpu().numpy()[0,:])
    print(tokenizer.decode(inputs['input_ids'].cpu().numpy()[0,:]))



############# local etn

@app.local_entrypoint()
def main():
    # healthcheck.remote()

    sanity_check_2 = exercise_hf_model_generate.remote()
    print(f"return obj:\n{sanity_check_2}")
