from typing import TYPE_CHECKING
import re
import torch

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from TransformerMTGP.util.singleton_module import pset


if TYPE_CHECKING:
    # from src.classes.operation import Operation
    from src.classes.operation_option import OperationOption


class BaseIndividual:
    def __init__(self) -> None:
        self.primitive_dict = {}

        for index, (k, v) in enumerate(pset.items()):
            # vector = torch.zeros(len(pset) + 1)
            vector = [0] * (len(pset) + 1)
            if v == "T":
                vector[0] = 1
            elif v == "F":
                vector[0] = 0
            vector[index + 1] = 1
            self.primitive_dict[k] = vector

    def t(self, obj: "OperationOption"):
        return 1 + 1

    def NIQ(self, obj: "OperationOption"):
        return obj.work_center.work_in_queue

    def WIQ(self, obj: "OperationOption"):
        return 1 + 1

    def MRT(self, obj: "OperationOption"):
        return 1 + 1

    def PT(self, obj: "OperationOption"):
        return obj.proc_time

    def NPT(self, obj: "OperationOption"):
        return obj.next_proc_time

    def LNPT(self, obj: "OperationOption"):
        return 1 + 1

    def MNPT(self, obj: "OperationOption"):
        return 1 + 1

    def DNPT(self, obj: "OperationOption"):
        return 1 + 1

    def ORT(self, obj: "OperationOption"):
        return 1 + 1

    def NRT(self, obj: "OperationOption"):
        return 1 + 1

    def WKR(self, obj: "OperationOption"):
        return obj.work_remaining

    def NOR(self, obj: "OperationOption"):
        return obj.num_ops_remaining

    def WINQ(self, obj: "OperationOption"):
        return 1 + 1

    def NINQ(self, obj: "OperationOption"):
        return 1 + 1

    def FDD(self, obj: "OperationOption"):
        return 1 + 1

    def DD(self, obj: "OperationOption"):
        return 1 + 1

    def W(self, obj: "OperationOption"):
        return obj.job.weight

    def AT(self, obj: "OperationOption"):
        return 1 + 1

    def MWT(self, obj: "OperationOption"):
        return self.system_time - obj.work_center.ready_time

    def OWT(self, obj: "OperationOption"):
        return self.system_time - obj.ready_time

    def NWT(self, obj: "OperationOption"):
        return 1 + 1

    def rFDD(self, obj: "OperationOption"):
        return 1 + 1

    def rDD(self, obj: "OperationOption"):
        return 1 + 1

    def IATM(self, obj: "OperationOption"):
        return 1 + 1

    def DJ(self, obj: "OperationOption"):
        return 1 + 1

    def LWR(self, obj: "OperationOption"):
        return 1 + 1

    def MWR(self, obj: "OperationOption"):
        return 1 + 1

    def AWR(self, obj: "OperationOption"):
        return 1 + 1

    def MNR(self, obj: "OperationOption"):
        return 1 + 1

    def NCM(self, obj: "OperationOption"):
        return 1 + 1

    def APTQ(self, obj: "OperationOption"):
        return 1 + 1

    def AWIS(self, obj: "OperationOption"):
        return 1 + 1

    def AOIS(self, obj: "OperationOption"):
        return 1 + 1

    def LWINQ(self, obj: "OperationOption"):
        return 1 + 1

    def MWINQ(self, obj: "OperationOption"):
        return 1 + 1

    def AWINQ(self, obj: "OperationOption"):
        return 1 + 1

    def LOINQ(self, obj: "OperationOption"):
        return 1 + 1

    def MOINQ(self, obj: "OperationOption"):
        return 1 + 1

    def AOINQ(self, obj: "OperationOption"):
        return 1 + 1

    def TWIS(self, obj: "OperationOption"):
        return 1 + 1

    def TOIS(self, obj: "OperationOption"):
        return 1 + 1

    def BT(self, obj: "OperationOption"):
        return 1 + 1

    def ABT(self, obj: "OperationOption"):
        return 1 + 1

    def NCJ(self, obj: "OperationOption"):
        return 1 + 1

    def TIS(self, obj: "OperationOption"):
        return self.system_time - obj.job.release_time

    def SL(self, obj: "OperationOption"):
        return 1 + 1

    def Max(self, a, b):
        return max(a, b)

    def Min(self, a, b):
        return min(a, b)

    def Sub(self, a, b):
        return a - b

    def Mul(self, a, b):
        return a * b

    def Div(self, a, b):
        if b == 0:
            return 1
        else:
            return a / b

    def Add(self, a, b):
        return a + b

    def parse_expression(self, expr):
        primitive_list = re.findall(r"\w+", expr)
        stack = []

        nodes = []
        x = []
        edge_index = []
        for k, primitive in enumerate(primitive_list):
            x.append(self.primitive_dict.get(primitive))
            nodes.append(
                {
                    "children": 0,
                    "name": primitive,
                    "terminal": (True if pset.get(primitive, "T") == "T" else False),
                    "index": k,
                }
            )

        for k, primitive_obj in enumerate(nodes):
            if not primitive_obj.get("terminal"):
                if len(stack) > 0:
                    parent = stack[-1]
                    parent["children"] += 1
                    if parent["children"] == 2:
                        stack.pop()
                    edge_index.append(torch.tensor((k, parent["index"])))
                stack.append(primitive_obj)
            elif primitive_obj.get("terminal"):
                if len(stack) > 0:
                    parent = stack[-1]
                    parent["children"] += 1
                    edge_index.append(torch.tensor((k, parent["index"])))
                    if parent["children"] == 2:
                        stack.pop()
                else:
                    # edge_index.append([])
                    # edge_index.append(torch.empty((0, 2), dtype=torch.long))
                    edge_index.append(torch.tensor((0, 0), dtype=torch.long))
        tensor_edge_index = torch.stack(edge_index).T
        # tensor_edge_index = torch.tensor(edge_index).T
        tensor_x = torch.tensor(x).to(torch.float32)
        return tensor_x, tensor_edge_index
