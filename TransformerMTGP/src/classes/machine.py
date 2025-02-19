from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from work_center import WorkCenter


class Machine:
    def __init__(self, id, work_center: "WorkCenter", ready_time: float = 0.0) -> None:
        self.__id = id
        self.work_center = work_center
        self.ready_time = ready_time
        self.bussy = False

    @property
    def id(self):
        return self.__id
