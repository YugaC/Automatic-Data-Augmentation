import yaml

configs = {}

def load_config(file_path="config.yaml"):
    global configs
    with open(file_path, 'r') as file:
        configs = yaml.safe_load(file)
    return configs 

def get_config(key):
    return configs.get(key)

# Load the default configuration initially
#load_config()

# Example usage
if __name__ == "__main__":
    load_config("/home/woody/iwi5/iwi5210h/Automatic-Data-Augmentation/config.yaml")  # Load default config for testing
    print(get_config("data_dir"))
    print(get_config("max_epochs"))
    print(get_config("root_dir"))
    print(get_config("val_interval"))
    print(get_config("data_dir_test"))
