import unittest

from work_center import WorkCenter
from job import Job
from operation import Operation
from operation_option import OperationOption
from machine import Machine

from individual import Individual

from process import Process


class TestMain(unittest.TestCase):

    def setUp(self) -> None:
        self.center_1 = WorkCenter(1, 2)

        self.c1_m1 = Machine(1, self.center_1)
        self.c1_m2 = Machine(2, self.center_1)

        self.job1 = Job(1, arrival_time=2, releaseTime=2, due_date=100, weight=10)
        self.j1_op1 = Operation(self.job1, 1, work_center=self.center_1)
        self.j1_op2 = Operation(self.job1, 2, work_center=self.center_1)

        self.j1_op1_o1 = OperationOption(self.j1_op1, 1, 32, self.center_1)
        self.j1_op1_o2 = OperationOption(self.j1_op1, 2, 15, self.center_1)

        self.j1_op2_o1 = OperationOption(self.j1_op2, 3, 10, self.center_1)
        self.j1_op2_o2 = OperationOption(self.j1_op2, 4, 40, self.center_1)

        self.process1 = Process(self.center_1, self.c1_m2.id, self.j1_op2_o2, 20.0)
        self.process2 = Process(self.center_1, self.c1_m1.id, self.j1_op1_o2, 10.0)

    def test_job(self):
        self.assertEqual(self.job1.id, 1)
        self.assertEqual(self.job1.arrival_time, 2)
        self.assertEqual(self.job1.release_time, 2)
        self.assertEqual(self.job1.weight, 10)
        self.assertEqual(self.job1.tardiness, 0)
        self.assertEqual(self.job1.weightedTardiness, 0)
        self.assertIsNone(self.job1.addOperation(self.j1_op1))
        self.assertIsNone(self.job1.addOperation(self.j1_op2))
        self.assertEqual(self.job1.getOperations(), [self.j1_op1, self.j1_op2])
        self.assertIsNone(self.job1.linkOperations())
        self.assertEqual(self.job1.getOperation(1).next, self.j1_op1)
        self.assertEqual(self.job1.getOperation(0).next, None)

        self.job1.__completion_time = 110

        self.assertEqual(self.job1.tardiness, 10)
        self.assertEqual(self.job1.weightedTardiness, 100)

    def test_operation(self):
        self.assertIsNone(self.job1.addOperation(self.j1_op1))
        self.assertIsNone(self.job1.addOperation(self.j1_op2))
        self.assertIsNone(self.job1.linkOperations())
        self.assertEqual(self.j1_op1.id, 1)
        self.assertEqual(self.j1_op2.next, self.j1_op1)

    def test_operation_option(self):
        self.assertIsNone(self.job1.addOperation(self.j1_op1))
        self.assertIsNone(self.job1.addOperation(self.j1_op2))
        self.assertIsNone(self.job1.linkOperations())

        self.assertEqual(self.j1_op1_o1.operation, self.j1_op1)
        self.assertEqual(self.j1_op2_o1.next, self.j1_op1)

        self.assertEqual(self.j1_op2_o1.job, self.job1)

        self.assertEqual(self.j1_op2_o1.prior_to(self.j1_op2_o1), False)

    def test_work_center(self):
        self.assertEqual(self.center_1.id, 1)
        self.assertEqual(self.center_1.process_time_in_queue(type="Max"), 0)

        self.assertIsNone(self.center_1.add_to_queue(self.j1_op1_o1))
        self.assertIsNone(self.center_1.add_to_queue(self.j1_op1_o2))
        self.assertIsNone(self.center_1.add_to_queue(self.j1_op2_o1))
        self.assertIsNone(self.center_1.add_to_queue(self.j1_op2_o2))
        self.assertEqual(self.center_1.process_time_in_queue(type="Min"), 10)
        self.assertEqual(self.center_1.num_ops_in_queue, 4)
        self.assertListEqual(
            self.center_1.remove_from_queue(self.j1_op1_o2),
            [self.j1_op1_o1, self.j1_op2_o1, self.j1_op2_o2],
        )
        self.assertEqual(self.center_1.work_in_queue, 82)

    def test_process(self):
        self.assertEqual(self.process1.duration, 40.0)
        self.assertEqual(self.process2.duration, 15.0)

        self.assertEqual(self.process1.compare_to(self.process2), 1)

    def test_parse_expression(self):
        individual = Individual(
            "Min(Sub(Max(Add(NPT(), TIS()), Mul(MWT(), NIQ())), Sub(Min(NOR(), TIS()), Div(NOR(), NIQ()))), Sub(Min(Div(TIS(), NOR()), Add(PT(), NPT())), Sub(Sub(TIS(), OWT()), Div(NPT(), PT()))))",
            "Div(Max(Mul(NOR(), Sub(W(), NIQ())), Sub(Min(TIS(), NOR()), Mul(NIQ(), MWT()))), Min(Add(Mul(NPT(), NOR()), Max(NPT(), NPT())), Max(Div(WKR(), PT()), Min(NOR(), OWT()))))",
        )
        # self.assertIsNone(
        #     individual.parse_expression("Min(Sub(Max(Add(NPT(), TIS()), Mul(MWT(), NIQ())), Sub(Min(NOR(), TIS()), Div(NOR(), NIQ()))), Sub(Min(Div(TIS(), NOR()), Add(PT(), NPT())), Sub(Sub(TIS(), OWT()), Div(NPT(), PT()))))")
        # )


if __name__ == "__main__":
    unittest.main()
