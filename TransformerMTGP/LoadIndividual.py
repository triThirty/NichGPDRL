import json
from TransformerMTGP.saveFile import formula_base_dir, base_dir


def load_individual_from_gen_json_format(config):
    path = formula_base_dir.substitute(**config)
    with open(
        path,
        "r",
    ) as fileName_individual:
        dict = json.load(fileName_individual)

    return dict


def load_individual_from_gen(config):
    path = base_dir.substitute(**config)
    with open(
        path,
        "r",
    ) as fileName_individual:
        dict = json.load(fileName_individual)
    return dict
