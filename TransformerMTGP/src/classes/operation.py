from typing import TYPE_CHECKING, Optional

# from operation_option import OperationOption


if TYPE_CHECKING:
    from work_center import WorkCenter
    from job import Job


class Operation:
    def __init__(
        self,
        job: "Job",
        id: int,
        proc_time: float = 0.0,
        work_center: Optional["WorkCenter"] = None,
    ) -> None:
        self.__job = job
        self.__id = id
        self.procTime = proc_time
        self.workCenter = work_center
        self.operation_options = []

        self.next = None
        # if work_center and proc_time:
        #     self.operation_options.append(OperationOption())

    @property
    def id(self):
        return self.__id

    @property
    def job(self):
        return self.__job

    def get_least_workload(self):
        raise NotImplementedError

    def chooseOperationOption(self, system, routing_rule):
        raise NotImplementedError


# if __name__ == "__main__":
#     testjob = job.Job(1, [], 2.0, 2.0, 3.0, 10.0)
#     operation = Operation(testjob, 1)
#     print(operation.workCenter)
