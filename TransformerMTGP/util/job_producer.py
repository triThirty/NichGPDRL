import random

from util.singleton_module import rng, settings
from src.classes.job import Job
from src.classes.operation import Operation
from src.classes.operation_option import OperationOption


class JobProducer:
    def __init__(self, num_work_center, min_ops, max_ops, util_level) -> None:
        self.__num_jobs_arrived = 0
        self.__current_time = 0

        self.__num_work_center = num_work_center
        self.__min_ops = min_ops
        self.__max_ops = max_ops
        self.__util_level = util_level

    def mean_arrival_time(
        self,
    ):
        mean_num_ops = 0.5 * (self.__min_ops + self.__max_ops)
        mean_proc_time = (
            50  # TODO: refinment this the calculation, use process_time_sampler
        )
        return (mean_num_ops * mean_proc_time) / (
            self.__util_level * self.__num_work_center
        )

    def process_time_sampler(self):
        return rng.uniform(1, 99)

    def job_weight_sampler(self):
        r = rng.uniform(0, 1)
        if r < 0.2:
            return 4
        elif r < 0.8:
            return 2
        else:
            return 1

    def job_arrival_time_sampler(self, mean):
        return rng.exponential(mean)

    def num_ops_sampler(self, min, max):
        return rng.integers(min, max)

    def job_producer(self):
        jobs = []
        current_time = 0
        num_jobs = random.randint(settings["num_jobs_min"], settings["num_jobs_max"])
        for i in range(num_jobs):
            job = {}
            job["id"] = str(i)
            current_time += self.job_arrival_time_sampler(5)
            job["arrival_time"] = current_time
            job["weight"] = self.job_weight_sampler()
            job["operations"] = []

            num_ops = self.num_ops_sampler(self.__min_ops, self.__max_ops)
            for i in range(num_ops):
                operation = {}
                operation["id"] = job["id"] + "-" + str(i)
                operation["operation_options"] = []
                num_op_options = self.num_ops_sampler(self.__min_ops, self.__max_ops)

                for j in range(num_op_options):
                    operation_option = {}
                    operation_option["id"] = operation["id"] + "-" + str(j)
                    operation_option["proc_time"] = self.process_time_sampler()
                    operation_option["work_center"] = random.randint(
                        0, self.__num_work_center - 1
                    )
                    operation["operation_options"].append(operation_option)
                job["operations"].append(operation)
            jobs.append(job)
        return jobs
