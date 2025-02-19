from typing import TYPE_CHECKING, Self

if TYPE_CHECKING:
    from operation import Operation
    from work_center import WorkCenter


class OperationOption:
    def __init__(
        self, operation: "Operation", id, proc_time, work_center: "WorkCenter"
    ) -> None:

        self.__operation = operation
        self.__id = id
        self.proc_time = proc_time
        self.work_center = work_center
        self.ready_time = 0.0
        self.work_remaining = 0.0
        self.num_ops_remaining = 0
        self.flow_due_date = 0.0
        self.next_proc_time = 0.0
        self.priority = 0.0

    @property
    def operation(self):
        return self.__operation

    @property
    def id(self):
        return self.__id

    @property
    def next(self):
        return self.__operation.next

    @property
    def job(self):
        return self.__operation.job

    def prior_to(self, option: Self) -> bool:
        if self.priority == option.priority:
            return True if self.id < option.id else False
        return True if self.priority > option.priority else False
