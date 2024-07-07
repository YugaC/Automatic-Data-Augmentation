import yaml

def get_config(key):
    with open("config.yaml", 'r') as file:
        configs = yaml.safe_load(file)
    return configs.get(key)

# Example usage
if __name__ == "__main__":
    print(get_config("data_dir"))
    print(get_config("max_epochs"))
    print(get_config("root_dir"))
    print(get_config("val_interval"))
    print(get_config("data_dir_test"))
