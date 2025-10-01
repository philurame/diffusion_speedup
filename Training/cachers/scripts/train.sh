export NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjMGQzNzAzMS1lMjQ2LTRhMjItOTIwNi0wMGYzNTg3ZTVlNTIifQ=="

python Training/cachers/train_reinforce.py \
    --config "configs/cachers/$METHOD.yaml" \
    --project_name "neudachina/reinforce-cacher" \
    --device "$DEVICE"

    # --config "configs/cachers/constant.yaml" \
    # --config "configs/cachers/mlp_true.yaml" \


# export NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjMGQzNzAzMS1lMjQ2LTRhMjItOTIwNi0wMGYzNTg3ZTVlNTIifQ=="

# python Training/cachers/train_reinforce.py \
#     --config "configs/cachers/constant.yaml" \
#     --project_name "neudachina/reinforce-cacher" \
#     --device "cuda:1"