# https://modal.com/apps/brendanchambers/main
#   modal tutorial 

uv add modal  # pip install modal
uv run python -m modal setup


# The web browser should have opened for you to authenticate and get an API token.
# If it didn't, please copy this URL into your web browser manually:

# https://modal.com/token-flow/tf-YT26936masqE3JNmBqNTxq

# Web authentication finished successfully!
# Token is connected to the brendanchambers workspace.
# Verifying token against https://api.modal.com
# Token verified successfully!
# Token written to /Users/bc/.modal.toml in profile brendanchambers

# sanity check:
uv run modal run src/demo/get_started.py
