import signal

from func_timeout import func_timeout, FunctionTimedOut
from pm4py.algo.simulation.playout.petri_net import algorithm as simulator
from pm4py.objects.log.obj import EventLog, Trace
from collections import Counter
import numpy as np


def playout_net(net, initial_marking, final_marking, timeout, type = ' extensive'):
    log = []

    try:
        if type=='extensive':
            parameters = {simulator.Variants.EXTENSIVE.value.Parameters.MAX_TRACE_LENGTH: 50}
            log = func_timeout(timeout, simulator.apply,
                               kwargs={'net': net, 'initial_marking': initial_marking, 'final_marking': final_marking,
                                       'variant': simulator.Variants.EXTENSIVE,
                                       'parameters': parameters})
        else:
            log = simulator.apply(net, initial_marking, final_marking,
                                            variant=simulator.Variants.BASIC_PLAYOUT, parameters={
                        simulator.Variants.BASIC_PLAYOUT.value.Parameters.NO_TRACES: 1000})

            # parameters = {simulator.Variants.BASIC_PLAYOUT.value.Parameters.NO_TRACES: 1000}
            # log = func_timeout(timeout, simulator.apply,
            #                    kwargs={'net': net, 'initial_marking': initial_marking, 'final_marking': final_marking,
            #                            'variant': simulator.Variants.BASIC_PLAYOUT,'parameters':parameters})

    except FunctionTimedOut:
        print("WARNING: Time out during trace computation.")
    except Exception as e:
        print('other exceptions.')
        print(e)

    log = filter_traces(log)
    case_id = 1
    for trace in log:
        trace.attributes["concept:name"] = "c" + str(case_id)
        case_id += 1
    return log


def filter_traces(log):
    cleaned_log = EventLog()
    variants = set()
    for trace in log:
        this_act_seqs = []
        for event in trace:
            this_act_seqs.append(event["concept:name"])
        this_act_seqs = tuple(this_act_seqs)

        if len(trace) > 1 and this_act_seqs not in variants and max(list(Counter(
                this_act_seqs).values())) <= 2:  # 1. removbe the duplicated traces.  2. loop is executed no more than 2.
            cleaned_log.append(trace)
            variants.add(this_act_seqs)

    return cleaned_log
