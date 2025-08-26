
import modal

# configure base image.  
# takes ~ few minutes to build unless cached
vllm_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "vllm==0.9.1",
        "huggingface_hub[hf_transfer]==0.32.0",
        "flashinfer-python==0.2.6.post1",
        extra_index_url="https://download.pytorch.org/whl/cu128",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})  # faster model transfers
)
app = modal.App(image=vllm_image)

# @app.function(gpu="T4:1")
# def healthcheck():
#     import torch
#     assert torch.cuda.is_available()

######## remote: exercise_pipeline
@app.function(gpu='T4:1')
def exercise_pipeline():

    from transformers import pipeline
    import torch

    # device=0, # "cuda" for Colab, "msu" for iOS devices
    pipe = pipeline(task="text-generation",
                    model="Qwen/Qwen3-0.6B-Base")
                    #device=0,
                    #torch_dtype=torch.bfloat16)
    messages = [
        {"role": "user", "content": "Who are you?"},
    ]
    result = pipe(messages)    
    print(result)  
    print('ep')

######## remote: excerise_hf_model_generate
@app.function(gpu='T4:1')
def exercise_hf_model_generate():
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B-Base")
    messages = [
        {"role": "user", "content": "Who are you?"},
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=40)
    print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))
    print('ehfmg')



############# local etn

@app.local_entrypoint()
def main():
    # healthcheck.remote()

    # sanity_check_1 = exercise_pipeline.remote()
    # print(f"return obj:\n{sanity_check_1}")
    
    sanity_check_2 = exercise_hf_model_generate.remote()
    print(f"return obj:\n{sanity_check_2}")

    # notice that small config differences lead to different outputs, understand apply_chat_template details

