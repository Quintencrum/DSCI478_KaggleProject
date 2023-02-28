from pathlib import Path


def get_project_path():
    return Path.cwd()

def get_data_path():
    # Need both src/data now because of file structure
    return Path.cwd().joinpath("src").joinpath("data")
