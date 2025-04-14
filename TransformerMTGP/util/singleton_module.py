import numpy as np
from typing import TYPE_CHECKING, Iterator


if TYPE_CHECKING:
    from src.classes.individual import Individual


def compute_phenotype(pop, decision_situations):
    for ind in pop:
        decision_vector = []
        for situation in decision_situations:
            selected_machine_index = GP_evolve_R(ind[0], *situation[0])
            decision_vector.append(selected_machine_index)
            job_position = GP_evolve_S(situation[1], ind[1])
            decision_vector.append(job_position)
        ind.decision_vector = decision_vector


# Set seed
# rng = np.random.default_rng(233)
rng = np.random.default_rng()


class Individuals(list):
    def __init_subclass__(cls) -> None:
        return super().__init_subclass__()

    def append(self, object: "Individual") -> None:
        return super().append(object)

    def __iter__(self) -> Iterator["Individual"]:
        return super().__iter__()


# pset = {
#     "t": "T",
#     "NIQ": "T",
#     "WIQ": "T",
#     "MRT": "T",
#     "PT": "T",
#     "NPT": "T",
#     "LNPT": "T",
#     "MNPT": "T",
#     "DNPT": "T",
#     "ORT": "T",
#     "NRT": "T",
#     "WKR": "T",
#     "NOR": "T",
#     "WINQ": "T",
#     "NINQ": "T",
#     "FDD": "T",
#     "DD": "T",
#     "W": "T",
#     "AT": "T",
#     "MWT": "T",
#     "OWT": "T",
#     "NWT": "T",
#     "rFDD": "T",
#     "rDD": "T",
#     "IATM": "T",
#     "DJ": "T",
#     "MWR": "T",
#     "LWR": "T",
#     "AWR": "T",
#     "MNR": "T",
#     "NCM": "T",
#     "APTQ": "T",
#     "AWIS": "T",
#     "AOIS": "T",
#     "LWINQ": "T",
#     "MWINQ": "T",
#     "AWINQ": "T",
#     "LOINQ": "T",
#     "MOINQ": "T",
#     "AOINQ": "T",
#     "TWIS": "T",
#     "TOIS": "T",
#     "BT": "T",
#     "ABT": "T",
#     "NCJ": "T",
#     "TIS": "T",
#     "SL": "T",
#     "Add": "F",
#     "Div": "F",
#     "Max": "F",
#     "Min": "F",
#     "Mul": "F",
#     "Sub": "F",
# }

pset = {
    # "t": "T",
    "NIQ": "T",
    "WIQ": "T",
    # "MRT": "T",
    "PT": "T",
    "NPT": "T",
    # "LNPT": "T",
    # "MNPT": "T",
    # "DNPT": "T",
    # "ORT": "T",
    # "NRT": "T",
    "WKR": "T",
    "NOR": "T",
    # "WINQ": "T",
    # "NINQ": "T",
    # "FDD": "T",
    # "DD": "T",
    # "W": "T",
    # "AT": "T",
    "MWT": "T",
    "OWT": "T",
    # "NWT": "T",
    # "rFDD": "T",
    # "rDD": "T",
    # "IATM": "T",
    # "DJ": "T",
    # "MWR": "T",
    # "LWR": "T",
    # "MWR": "T",
    # "AWR": "T",
    # "MNR": "T",
    # "NCM": "T",
    # "APTQ": "T",
    # "AWIS": "T",
    # "AOIS": "T",
    # "LWINQ": "T",
    # "MWINQ": "T",
    # "AWINQ": "T",
    # "LOINQ": "T",
    # "MOINQ": "T",
    # "AOINQ": "T",
    # "TWIS": "T",
    # "TOIS": "T",
    # "BT": "T",
    # "ABT": "T",
    # "NCJ": "T",
    "TIS": "T",
    "SLACK": "T",
    "add": "F",
    "protected_div": "F",
    "maximum": "F",
    "minimum": "F",
    "multiply": "F",
    "subtract": "F",
}


existing_individuals = Individuals()

settings = {
    "num_jobs_min": 5,
    "num_jobs_max": 100,
    "num_work_center": 2,
    "min_ops": 1,
    "max_ops": 10,
    "util_level": 1,
    "learning_rate": 0.1,
    "max_iterations": 100,
}
