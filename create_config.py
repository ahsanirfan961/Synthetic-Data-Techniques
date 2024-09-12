import yaml, argparse


parser = argparse.ArgumentParser(description='Variables for a yaml file')
parser.add_argument('--hf_token', type=str, required=True)
parser.add_argument('--input_dataset_path', type=str, required=True)
parser.add_argument('--output_dataset_path', type=str, required=True)
parser.add_argument('--input_batch_size', type=int, required=True)
parser.add_argument('--new_max_tokens', type=int, required=True)
parser.add_argument('--temperature', type=float, required=True)
parser.add_argument('--instruct_model_path', type=str, required=True)
parser.add_argument('--response_model_path', type=str, required=True)

args = parser.parse_args()

data = {
    "hf-token": args.hf_token,
    "datasets": [
        {
            "path": args.input_dataset_path,
            "type": "input"
        },
        {
            "path": args.output_dataset_path,
            "type": "output"
        }
    ],
    "models": [
        {
            "path": args.instruct_model_path,
            "type": "instruct"
        },
        {
            "path": args.response_model_path,
            "type": "response"
        }
    ],
    "input-batch-size": args.input_batch_size,
    "max-tokens": args.new_max_tokens,
    "temperature": args.temperature,
}

with open('temp.yaml', 'w') as file:
    yaml.dump(data, file)