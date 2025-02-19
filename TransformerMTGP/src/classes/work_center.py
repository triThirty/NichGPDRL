from typing import List, Literal, TYPE_CHECKING

from src.classes.machine import Machine

if TYPE_CHECKING:
    from operation_option import OperationOption
    from machine import Machine


class WorkCenter:
    def __init__(
        self,
        id,
        num_machines,
        queue: List["OperationOption"] | None = None,
        work_in_queue=0,
        bussy_time=0,
    ) -> None:
        self.__id = id
        self.num_machines = num_machines
        self.__queue = [] if queue is None else queue
        self.machine_ready_times = {str(i): 0 for i in range(num_machines)}
        self.work_in_queue = work_in_queue
        self.bussy_time = bussy_time
        self.machines = {}
        for i in range(num_machines):
            self.machines[str(i)] = Machine(i, self)
            # self.machines.append(Machine(i, self))

    @property
    def id(self):
        return self.__id

    @property
    def queue(self):
        return self.__queue

    def process_time_in_queue(self, type: Literal["Max", "Min"]):
        if self.__queue == []:
            return 0
        else:
            if type == "Min":
                return min(self.__queue, key=lambda x: x.proc_time).proc_time
            else:
                return max(self.__queue, key=lambda x: x.proc_time).proc_time

    @property
    def ready_time(self):
        return min(self.machine_ready_times.values())

    @property
    def num_ops_in_queue(self):
        return len(self.__queue)

    def reset(self, ready_time: float = 0.0):
        self.__queue.clear()
        self.machine_ready_times = {str(i): 0 for i in range(self.num_machines)}
        self.work_in_queue = 0.0
        self.bussy_time = ready_time
        for machine in self.machines.values():
            machine.ready_time = ready_time
            machine.bussy = False

    @property
    def earliest_ready_machine(self):
        machine_id = min(self.machine_ready_times, key=self.machine_ready_times.get)
        return self.machines[int(machine_id)]

        # return Machine(index, self, earlist_ready_time)

    def add_to_queue(self, opo: "OperationOption"):
        self.__queue.append(opo)
        self.work_in_queue += opo.proc_time

    def remove_from_queue(self, opo: "OperationOption"):
        self.__queue.remove(opo)
        if len(self.__queue) == 0:
            self.work_in_queue = 0
        else:
            self.work_in_queue -= opo.proc_time
        return self.__queue

    @property
    def unbussy_machines(self):
        unbussy_machines = [
            machine for machine in self.machines.values() if not machine.bussy
        ]
        return True if unbussy_machines else False, unbussy_machines


# if __name__ == "__main__":
#     test_worker_center = WorkCenter(1, 2)
#     test_worker_center.process_time_in_queue("Max")
