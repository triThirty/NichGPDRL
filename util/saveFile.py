import json
from string import Template
from util.decorator import ensure_directory_exists


root_dir = r"./data/${algo}_${path_surfix}/scenario_${scenarios}"

formula_base_dir = Template(
    f"{root_dir}/${{seeds}}_meng_individual_${{scenarios}}_formula_format.json"
)
base_dir = Template(f"{root_dir}/${{seeds}}_meng_individual_${{scenarios}}.json")
txt_base_dir = Template(f"{root_dir}/${{seeds}}_${{scenarios}}_each_gen.txt")


@ensure_directory_exists(formula_base_dir)
def save_each_gen_best_individual_json_format(config, best_ind_all_gen):
    individual_dict = []

    for key, ind in enumerate(best_ind_all_gen):
        individual_dict.append(
            {
                "T0": str(ind[0]),
                "T1": str(ind[1]),
                "fitness": 0,
            }
        )

    path = formula_base_dir.substitute(**config)
    with open(
        path,
        "w",
    ) as fileName_individual:
        json.dump(individual_dict, fileName_individual)

    return


@ensure_directory_exists(formula_base_dir)
def save_each_gen_best_individual_on_test_dataset(config, best_ind_all_gen_dict):
    path = formula_base_dir.substitute(**config)
    with open(
        path,
        "w",
    ) as fileName_individual:
        json.dump(best_ind_all_gen_dict, fileName_individual)


@ensure_directory_exists(base_dir)
def save_each_gen_best_individual_meng(config, best_ind_all_gen):
    individual_dict = []

    for gen in range(len(best_ind_all_gen)):
        best_ind = best_ind_all_gen[gen]

        if len(best_ind) == 2:
            sequencing = best_ind[0]
            routing = best_ind[1]
        else:
            sequencing = best_ind[0]

        individual = []
        sequencing_list = []
        for i in range(len(sequencing)):
            sequencing_list.append(sequencing[i].name)

        if len(best_ind) == 2:
            routing_list = []
            for i in range(len(routing)):
                routing_list.append(routing[i].name)

        individual.append(sequencing_list)
        if len(best_ind) == 2:
            individual.append(routing_list)

        individual_dict.append(individual)

    path = base_dir.substitute(**config)
    with open(
        path,
        "w",
    ) as fileName_individual:
        json.dump(individual_dict, fileName_individual)


@ensure_directory_exists(txt_base_dir)
def clear_individual_each_gen_to_txt(config):
    path = txt_base_dir.substitute(**config)
    with open(
        path,
        "w",
    ) as file:
        file.write("Best individuals from each gen:\n")
    return


@ensure_directory_exists(txt_base_dir)
def save_individual_each_gen_to_txt(config, individuals, gen):
    path = txt_base_dir.substitute(**config)
    with open(
        path,
        "a",
    ) as file:
        file.write("\nGen: " + str(gen) + "\n")
        file.write("Individual:\n")
        file.write("Tree 0:\n")  # routing rule
        file.write(str(individuals[0]) + "\n")
        file.write("Tree 1:\n")  # sequencing rule
        file.write(str(individuals[1]) + "\n")
    return
