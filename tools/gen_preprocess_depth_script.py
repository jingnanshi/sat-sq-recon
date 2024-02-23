import csv


def load_first_column(csv_file_path):
    first_column = []

    with open(csv_file_path, 'r', newline='') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            if row:  # This checks if the row is not empty
                first_column.append(row[0])  # Append the first column value to the list
    return first_column


def generate_shell_script(data, output_file_path, base_command=""):
    with open(output_file_path, 'w') as file:
        # Loop through each item in the list and write it to the shell script
        for item in data:
            file.write(f'{base_command}{item}\n')


if __name__ == "__main__":
    #basecmd = "python tools/preprocess_depth.py --cfg experiments/config.yaml --model_name="
    basecmd = "--model_name="
    sh_path = "tools/preprocess_depth_all_objs.sh"

    model_names = load_first_column('/mnt/jnshi_data/datasets/hydra_objects_data/spe3r/splits.csv')
    generate_shell_script(model_names, sh_path, basecmd)
    print(f"Bash file written to {sh_path}.")
