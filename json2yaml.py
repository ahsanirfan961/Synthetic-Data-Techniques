import json
import yaml
import argparse

def convert_json_to_yaml(input_path, output_path):
    # Read JSON file
    with open(input_path, 'r') as json_file:
        json_data = json.load(json_file)

    # Convert JSON data to YAML
    yaml_data = yaml.dump(json_data, sort_keys=False)

    # Write YAML data to file
    with open(output_path, 'w') as yaml_file:
        yaml_file.write(yaml_data)

    print(f"Conversion completed! YAML file saved as '{output_path}'.")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Convert JSON file to YAML.')
    parser.add_argument('--input', help='Path to the input JSON file')
    parser.add_argument('--output', help='Path to the output YAML file')

    # Parse arguments
    args = parser.parse_args()

    # Convert JSON to YAML
    convert_json_to_yaml(args.input, args.output)

if __name__ == '__main__':
    main()
