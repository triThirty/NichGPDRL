import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from collections import deque
import re
from typing import TYPE_CHECKING

from src.classes.base_individual import BaseIndividual
from src.classes.job import Job
from src.classes.operation import Operation
from src.classes.operation_option import OperationOption
from util.singleton_module import settings

if TYPE_CHECKING:
    from src.classes.work_center import WorkCenter
    from src.classes.machine import Machine


class Individual(BaseIndividual):
    def __init__(self, sequence: str, route: str, work_centers=[], id=0) -> None:
        super().__init__()
        self.__id = id
        self.__sequence = sequence
        self.__route = route
        self.__fitness = 0
        self.__true_fitness = 0
        self.__num_iteration = 0
        self.evaluated = False
        self.sequence_func = self.compiler(sequence)
        self.route_func = self.compiler(route)
        self.sequence_data, self.sequence_edge = self.parse_expression(sequence)
        self.route_data, self.route_edge = self.parse_expression(route)

        self.__jobs = {}

        self.__work_centers = work_centers
        self.__event_queue = deque()

        self.__system_time = 0

        self.fitness_list = []

    def generate_jobs_from_dict(self, jobs):
        self.__jobs = jobs
        for job in jobs:
            j = Job(
                job["id"],
                None,
                job["arrival_time"],
                job["arrival_time"],
                0,
                job["weight"],
            )
            for op in job["operations"]:
                operation = Operation(j, op["id"])
                for opo in op["operation_options"]:
                    operation_option = OperationOption(
                        operation,
                        opo["id"],
                        opo["proc_time"],
                        self.__work_centers[str(opo["work_center"])],
                    )
                    operation.operation_options.append(operation_option)
                j.add_operation(operation)
            j.link_operations()
            self.__event_queue.append(j)

    @property
    def id(self):
        return self.__id

    @property
    def true_fitness(self):
        return self.__true_fitness

    @true_fitness.setter
    def true_fitness(self, value):
        self.__true_fitness = value

    @property
    def system_time(self):
        return self.__system_time

    @property
    def sequence(self):
        return self.__sequence

    @property
    def work_centers(self):
        return self.__work_centers

    @property
    def route(self):
        return self.__route

    def compiler(self, expression: str):
        expr_str = re.sub(r"(\w+)", r"self.\1", expression)
        expr_str = re.sub(r"\(\)", "(obj)", expr_str)

        def func(self=self, obj=None):
            return eval(expr_str)

        return func

    def evaluate_sequence_priority(self, work_center: "WorkCenter"):
        if work_center.num_ops_in_queue != 0:
            for option in work_center.queue:
                option.priority = self.sequence_func(obj=option)
            work_center.queue.sort(key=lambda x: x.priority, reverse=True)
            option = work_center.queue.pop(0)
            return option
        else:
            return None

    def evaluate_routing_priority(self, op) -> OperationOption:
        for option in op.operation_options:
            option.priority = self.route_func(obj=option)
        best_operation_option = max(op.operation_options, key=lambda x: x.priority)
        best_operation_option.work_center.add_to_queue(best_operation_option)
        return best_operation_option

    def __update(self, option: "OperationOption"):
        job_is_finished, job_fitness_value = option.job.finish_operation(
            option, self.system_time
        )
        if job_is_finished:
            self.__fitness += job_fitness_value

    def __executer(self, executed_option, machine: "Machine"):
        machine.bussy = True
        work_center = executed_option.work_center

        machine.ready_time = self.system_time + executed_option.proc_time
        work_center.machine_ready_times[machine.id] = (
            self.system_time + executed_option.proc_time
        )

        self.__event_queue.append(
            {
                "work_center": str(work_center.id),
                "machine": str(machine.id),
                "operation_option": executed_option,
                "arrival_time": self.system_time + executed_option.proc_time,
            }
        )

    def check_work_center_queue(self, option: "OperationOption"):
        has_unbusy_machines, unbussy_machines = option.work_center.unbussy_machines
        if has_unbusy_machines:
            i = min(option.work_center.num_ops_in_queue, len(unbussy_machines))
            for j in range(i):
                machine = unbussy_machines[j]
                executed_option = self.evaluate_sequence_priority(option.work_center)
                if executed_option is not None:
                    self.__executer(executed_option, machine)

    def __reset(self):
        self.__system_time = 0.0
        self.__fitness = 0.0
        for work_center in self.__work_centers.values():
            work_center.reset(self.system_time)

    def controler(self):
        while len(self.__event_queue) > 0:
            self.__event_queue = deque(
                sorted(
                    self.__event_queue,
                    key=lambda x: (
                        x.arrival_time if isinstance(x, Job) else x["arrival_time"]
                    ),
                )
            )
            event = self.__event_queue.popleft()
            self.__system_time = (
                event.arrival_time if isinstance(event, Job) else event["arrival_time"]
            )
            if isinstance(event, Job):
                op = event.get_operations[0]
                operation = self.evaluate_routing_priority(op)
                self.check_work_center_queue(operation)
            elif isinstance(event, dict):
                work_center_id = event["work_center"]
                machine_id = event["machine"]
                work_center = self.__work_centers[str(work_center_id)]
                machine = work_center.machines[machine_id]
                machine.bussy = False
                self.__update(event["operation_option"])
                next_op = event["operation_option"].next
                if next_op is not None:
                    next_operation = self.evaluate_routing_priority(next_op)
                    self.check_work_center_queue(next_operation)
                executed_option = self.evaluate_sequence_priority(work_center)
                if executed_option is not None:
                    self.__executer(executed_option, work_center.machines[machine_id])
        self.__fitness = self.__fitness / len(self.__jobs)
        self.__num_iteration += 1
        self.__true_fitness = self.__true_fitness - settings["learning_rate"] * (
            0.999
        ) ** self.__num_iteration * (self.__true_fitness - self.__fitness)

        print(
            f"Individual: {self.__id}, finished {self.__num_iteration} evmulations \n"
        )
        # self.fitness_list.append(self.__true_fitness)
        # if self.__num_iteration % 100 == 0:
        #     import matplotlib.pyplot as plt

        #     plt.plot(self.fitness_list)
        #     plt.xlabel("Iteration")
        #     plt.show()
        #     print(self.fitness_list)
        self.__reset()
