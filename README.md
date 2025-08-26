
# minimal modal example

This differs slightly from the official modal example by using uv.  

## set up modal and run sanity check for cpu container

```
bash scripts/setup.sh
```

## sanity check a gpu container from cuda docker image + sample additional reqs

At this point I also updated the modal image builder to latest version:  https://modal.com/settings/brendanchambers/image-config

```
uv run python -m modal run src/explore/hf_inference_demo.py
```

For this example, 
- first build + run: ~12m  
- cached build + run: ~2.5m
- cached build + cached model + run: 

- build from cache ~1s
- load tokenizer and model (from hf cache? iiuc) ~15s
- generate 40 tokens ~