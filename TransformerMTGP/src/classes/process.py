from typing import TYPE_CHECKING, Self

if TYPE_CHECKING:
    from work_center import WorkCenter
    from operation_option import OperationOption


class Process:
    def __init__(
        self,
        work_center: "WorkCenter",
        machine_id: int,
        opo: "OperationOption",
        start_time: float,
    ) -> None:
        self.work_center = work_center
        self.machine_id = machine_id
        self.opo = opo
        self.__start_time = start_time
        self.__finish_time = self.__start_time + self.opo.proc_time

    @property
    def duration(self):
        return self.__finish_time - self.__start_time

    @property
    def start_time(self):
        return self.__start_time

    @property
    def finish_time(self):
        return self.__finish_time

    def compare_to(self, process: Self):
        if self.__start_time < process.__start_time:
            return -1
        elif self.__start_time > process.__start_time:
            return 1
        else:
            return 0
