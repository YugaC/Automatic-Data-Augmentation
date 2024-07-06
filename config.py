def get_config(key):
    configs = {
        "data_dir":'C:/Users/Yugashree/subset',
        "max_epochs": '600',
        "root_dir": 'C:/Users/Yugashree/Automatic-Data-Augmentation',
        "val_interval": 2,
        "data_dir_test": 'C:/Users/Yugashree/Downloads/subset/debugging',

    }
    return configs[key]