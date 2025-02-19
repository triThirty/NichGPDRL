from concurrent.futures import ThreadPoolExecutor

from util.singleton_module import existing_individuals, settings
from util.job_producer import JobProducer
import json


class DynamicSimulation:
    def __init__(self) -> None:
        self.__num_generated_jobs = 0
        self.__jp = JobProducer(
            settings["num_work_center"],
            settings["min_ops"],
            settings["max_ops"],
            settings["util_level"],
        )

    @property
    def num_generated_jobs(self):
        return self.__num_generated_jobs

    def run(self):
        iterations = 0
        while iterations < settings["max_iterations"]:
            jobs = self.__jp.job_producer()
            for indi in existing_individuals:
                indi.generate_jobs_from_dict(jobs)
            with ThreadPoolExecutor(max_workers=8) as executor:
                executor.map(lambda x: x.controler(), existing_individuals)

            with open("results/result_ind.json", "r") as f:
                data = json.load(f)
            for ind in existing_individuals:
                data[ind.id]["fitness"] = ind.true_fitness
            with open("results/result_ind.json", "w") as f:
                json.dump(data, f, indent=4)
            iterations += 1
