import statistics

from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from operation import Operation
    from operation_option import OperationOption


class Job:
    def __init__(
        self,
        id: int,
        operations: List["Operation"] | None = None,
        arrival_time: float = 0.0,
        releaseTime: float = 0.0,
        due_date: float = 0.0,
        weight: float = 0.0,
    ) -> None:
        self.__id = id
        self.__operations = [] if operations is None else operations
        self.__arrival_time = arrival_time
        self.__releaseTime = releaseTime
        self.__due_date = due_date
        self.__weight = weight

        self.__num_remaining_ops = len(self.__operations)

        self.__flow_time = 0.0

        self.__completion_time = 0.0
        self.__total_proc_time = 0.0
        self.__avg_proc_time = 0.0

    @property
    def id(self):
        return self.__id

    @property
    def arrival_time(self):
        return self.__arrival_time

    @property
    def release_time(self):
        return self.__releaseTime

    @property
    def flow_time(self):
        return self.__flow_time

    @property
    def weight(self):
        return self.__weight

    @property
    def tardiness(self):
        return (
            self.__completion_time - self.__due_date
            if self.__completion_time - self.__due_date >= 0
            else 0
        )

    @property
    def num_remaining_ops(self):
        return self.__num_remaining_ops

    @property
    def weightedTardiness(self):
        return self.weight * self.tardiness

    @property
    def get_operations(self):
        return self.__operations

    def add_operation(self, operation: "Operation"):
        self.__num_remaining_ops += 1
        self.__operations.append(operation)

    def get_operation(self, idx):
        return self.__operations[idx]

    def link_operations(self):
        next_operation = None
        next_proc_time = 0
        work_remaining_time = 0
        for k, operation in enumerate(self.__operations[::-1]):
            operation.next = next_operation
            next_operation = operation
            median_proc_time = statistics.median(
                [option.proc_time for option in operation.operation_options]
            )
            for option in operation.operation_options:
                option.num_ops_remaining = k
                option.next_proc_time = next_proc_time
                option.work_remaining = work_remaining_time + median_proc_time
            next_proc_time = median_proc_time
            work_remaining_time += median_proc_time

    def finish_operation(self, option: "OperationOption", system_time: float):
        self.__total_proc_time += option.proc_time
        self.__num_remaining_ops -= 1
        if self.__num_remaining_ops == 0:
            self.__completion_time = system_time
            self.__flow_time = self.__completion_time - self.arrival_time
            self.__avg_proc_time = self.__total_proc_time / len(self.__operations)
            return True, self.__flow_time - self.__total_proc_time
        else:
            return False, 0

    @property
    def completion_time(self):
        return self.__completion_time

    @property
    def total_proc_time(self):
        return self.__total_proc_time
