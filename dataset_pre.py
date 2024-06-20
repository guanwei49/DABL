import copy
import gc
import json
import os
import random
from pathlib import Path

from processmining.log import EventLog
import pandas as pd
from tqdm import tqdm
from pm4py.visualization.petri_net.common import visualize
from pm4py.visualization.petri_net import visualizer as pn_visualizer
from conversion.BPMN_parser import BpmnModelParser
from conversion.filters import DataFilter
from conversion.detector import ModelLanguageDetector
from conversion.jsontopetrinetconverter import JsonToPetriNetConverter
from pm4py.algo.simulation.playout.petri_net import algorithm as simulator
import pickle
from pm4py.algo.analysis.woflan import algorithm as woflan
import pm4py
import conversion.petrinetanalysis as pna
from generation.anomaly import *
from pm4py.objects.process_tree.obj import Operator
from pm4py.objects.conversion.process_tree import converter as pt_converter


## path of downloaded process model dataset
base_dir = './dataset/process_model'


BPMAIdataPath = os.path.join(base_dir,'BPMAI','models')
FBPMdataPath = os.path.join(base_dir,'FBPM2-ProcessModels')
SAPdataPath = os.path.join(base_dir,'sap_sam_2022','models')

petri_dir = os.path.join(base_dir, 'petri')  # contains petri net for process models
exclusive_dir = os.path.join(base_dir, 'exclusive')  # conmtains exclusive part in petri net
exclusive_anomalous_petri_dir = os.path.join(base_dir,
                                             'exclusive_anomalous_petri')  # conmtains petri net that the exclusive operator is modified to parallel.

model_log_dir = os.path.join(base_dir, 'model_log')
anomalous_model_log_dir = os.path.join(base_dir, 'anomalous_model_log')
act_name_path = os.path.join(base_dir, 'act_name.json')

playout_timeout = 6

if not os.path.exists(petri_dir):
    os.makedirs(petri_dir)
if not os.path.exists(exclusive_dir):
    os.makedirs(exclusive_dir)
if not os.path.exists(exclusive_anomalous_petri_dir):
    os.makedirs(exclusive_anomalous_petri_dir)
if not os.path.exists(model_log_dir):
    os.makedirs(model_log_dir)
if not os.path.exists(anomalous_model_log_dir):
    os.makedirs(anomalous_model_log_dir)


def _is_relevant_label(task_name):
    terms = {"message", "task"}
    if task_name is None:
        return True
    # if task_name == None:
    #     return False
    if task_name == "":
        return False
    if task_name.isnumeric():
        return False
    if task_name.lower() in terms:
        return False
    if "Gateway" in task_name:
        return False
    # if task_name.startswith("Exclusive_Databased_Gateway") \
    #         or task_name.startswith("EventbasedGateway") \
    #         or task_name.startswith("ParallelGateway") \
    #         or task_name.startswith("InclusiveGateway"):
    #     return False
    if task_name.startswith("EventSubprocess") or task_name.startswith("Subprocess"):
        return False
    return True


def cleanse_tree(tree):
    '''
        remove irrelevant node (such as 'Gateway';'EventSubprocess',''. irrelevant label is definded in function '_is_relevant_label')
    '''
    valid_children = []
    for node in tree.children:
        if len(node.children) == 0:
            if _is_relevant_label(node.label):
                if node.label is not None:
                    node.label = ' '.join(node.label.split())  # remove space in activity names
                valid_children.append(node)
        else:
            cleanse_tree(node)
            if len(node.children) > 0:
                valid_children.append(node)
    tree.children = valid_children
    # print(tree)


def get_leaves(root):
    if len(root.children) == 0:
        if root.label is not None:
            return [root.label]
        else:
            return []

    leaves = []
    stack = [root]  # 初始化堆栈，根节点入栈

    while stack:
        node = stack.pop()  # 弹出栈顶节点
        if len(node.children) == 0:
            if node.label is not None:
                leaves.append(node.label)

        # 将当前节点的子节点逆序压入堆栈，保证先访问左子节点再访问右子节点
        for child in reversed(node.children):
            stack.append(child)
    return leaves


def get_exclusvies(tree):
    exclusiveNodes = []

    stack = [tree]  # 初始化堆栈，根节点入栈
    while stack:
        node = stack.pop()  # 弹出栈顶节点

        if node.operator is not None and node.operator == Operator.XOR and node != tree:  # 是eclusive节点
            exclusiveNodes.append(node)

        # 将当前节点的子节点逆序压入堆栈，保证先访问左子节点再访问右子节点
        for child in reversed(node.children):
            stack.append(child)
    # get_leaves(exclusiveNodes[0].children[2])
    exclusives = []
    exclusiveNodes_real = []  # 去掉了那些 有一个空节点，一个正常节点的 node  （例如 ： X( tau, 'book attraction')）
    for exclusiveNode in exclusiveNodes:
        exclusive = []
        for node in exclusiveNode.children:
            acts = get_leaves(node)
            if len(acts) > 0:  # 为0时，是空节点 （tau）
                exclusive.append(get_leaves(node))
        if len(exclusive) > 1:
            exclusives.append(exclusive)
            exclusiveNodes_real.append(exclusiveNode)
    return exclusives, exclusiveNodes_real


def to_petri_helper(tree, case_name):
    if len(tree.children) == 0:  # only have one node， useless
        return False

    cleanse_tree(tree)

    acts = get_leaves(tree)

    if len(acts) > 0 and np.mean([len(act) for act in
                                  acts]) > 2:  # if activity name is meaningless character such as 'A', 'B', then we filter this petriNet.
        net, initial_marking, final_marking = pt_converter.apply(tree,
                                                                 variant=pt_converter.Variants.TO_PETRI_NET)
        pnet_file = os.path.join(petri_dir, case_name + ".pnser")
        pickle.dump((net, initial_marking, final_marking), open(pnet_file, 'wb'))

        exclusvies, exclusiveNodes = get_exclusvies(
            tree)  # exclusvies存储activity name; exclusiveNodes存储exclusive（XOR）节点
        # pm4py.view_process_tree(tree, format='png')

        gviz = visualize.apply(net, initial_marking, final_marking, parameters={"format": "png"})
        pn_file = os.path.join(petri_dir, case_name + ".png")
        pn_visualizer.save(gviz, pn_file)

        # print(exclusvies)
        exclusive_file = os.path.join(exclusive_dir, case_name + ".json")
        with open(exclusive_file, 'w') as f:
            json.dump(exclusvies, f)

        for ith, exclusiveNode in enumerate(exclusiveNodes):  # 构建exclusive异常的petri
            exclusiveNode.operator = Operator.PARALLEL
            net, initial_marking, final_marking = pt_converter.apply(tree,
                                                                     variant=pt_converter.Variants.TO_PETRI_NET)

            pnet_file = os.path.join(exclusive_anomalous_petri_dir, f"{case_name}_{ith}.pnser")
            pickle.dump((net, initial_marking, final_marking), open(pnet_file, 'wb'))
            exclusiveNode.operator = Operator.XOR

        return True
    else:
        return False


def convert_BPMAI_jsons_to_petri():
    converter = JsonToPetriNetConverter()
    json_files = os.listdir(BPMAIdataPath)
    json_files_en = []
    for json_file in tqdm(json_files):
        sp = json_file.split('.')
        if len(sp) == 3 and sp[-1] == 'json':
            with open(os.path.join(BPMAIdataPath, json_file)) as f:
                data = json.load(f)
                if data['model']['naturalLanguage'] == 'en' and 'BPMN' in data['model']['groupName']:
                    json_files_en.append(sp[0] + '.json')

    json_files_en.sort()
    print("Total number of json files in English:", len(json_files_en))
    success = 0
    failed = 0

    for json_file in json_files_en:
        case_name = os.path.basename(json_file).split('.')[0]
        tree = None
        try:
            # Load and convert json-based BPMN into Petri net
            net, initial_marking, final_marking = converter.convert_to_petri_net(
                os.path.join(BPMAIdataPath, json_file))

            # from pm4py.visualization.petri_net import visualizer as pn_visualizer
            # gviz = pn_visualizer.apply(net, initial_marking, final_marking)
            # pn_visualizer.view(gviz)

            # simulated_log = simulator.apply(net, initial_marking, final_marking, variant=simulator.Variants.EXTENSIVE, parameters={
            #     simulator.Variants.EXTENSIVE.value.Parameters.MAX_TRACE_LENGTH: 50})
            # simulated_log = simulator.apply(net,  initial_marking, final_marking, variant=simulator.Variants.BASIC_PLAYOUT, parameters={
            #     simulator.Variants.BASIC_PLAYOUT.value.Parameters.NO_TRACES: 50})

            tree = pm4py.convert_to_process_tree(net, initial_marking, final_marking)

            # pm4py.view_process_tree(tree, format='png')
        except:
            print("WARNING: Error during conversion from bpmn to Petri net.")
            failed += 1

        if tree is not None:
            flag = to_petri_helper(tree, case_name)

            if flag:
                success += 1
            else:
                failed += 1

        print('convert_BPMAI_jsons_to_petri:', success + failed, "jsons done. Succes: ", success, "failed: ", failed)
        if (success + failed) % 50 == 0:
            gc.collect()


# ll=[]
# for case in simulated_log:
#     ss = []
#     for event in case:
#         ss.append(event['concept:name'])
#     ll.append(ss)


def convert_FBPM_BPMN_to_petri():
    chapters = os.listdir(FBPMdataPath)

    success = 0
    failed = 0

    for chapter in chapters:
        if 'Chapter' in chapter:
            bpmn_files = os.listdir(os.path.join(FBPMdataPath, chapter))
            for bpmn in bpmn_files:
                case_name = os.path.basename(bpmn).split('.')[0]
                tree = None
                try:
                    bpmn_graph = pm4py.read_bpmn(os.path.join(FBPMdataPath, chapter, bpmn))
                    net, initial_marking, final_marking = pm4py.convert_to_petri_net(bpmn_graph)

                    tree = pm4py.convert_to_process_tree(net, initial_marking, final_marking)

                except Exception:
                    print("WARNING: Error during conversion from bpmn to Petri net:",
                          os.path.join(FBPMdataPath, chapter, bpmn))
                    failed += 1

                if tree is not None:
                    flag = to_petri_helper(tree, case_name)

                    if flag:
                        success += 1
                    else:
                        failed += 1

                print('convert_FBPM_BPMN_to_petri:', success + failed, "jsons done. Succes: ", success, "failed: ",
                      failed)
                if (success + failed) % 50 == 0:
                    gc.collect()


def convert_SAP_jsons_to_petri():
    CSV_files = sorted(Path(SAPdataPath).glob('*.csv'))
    p = BpmnModelParser()  # comment this line if you have already created the pkl file
    df_bpmn = p.parse_model_elements(CSV_files)
    # df_bpmn = DataFilter(df_bpmn).filter_data("example_processes_bpmn")

    df_bpmn = DataFilter(df_bpmn).filter_data("models",
                                              1)  # Filtering out models with with less than 1 elements and no start, end, or task elements...

    ld = ModelLanguageDetector(0.8)  # comment this line if you have already created the pkl file
    df_bpmn = ld.get_detected_natural_language_from_bpmn_model(
        df_bpmn)  # comment this line if you have already created the pkl file
    valid_model_id = df_bpmn[df_bpmn.detected_natural_language == 'en'].index  # find model in English

    print('number of valid model_id:', len(valid_model_id))
    with open('valid_modex_id_SAP.json', 'w') as f:
        json.dump(list(valid_model_id), f)

    # with open('valid_modex_id_SAP.json') as f:
    #     valid_model_id = json.load(f)

    success = 0
    failed = 0
    converter = JsonToPetriNetConverter()

    for file in CSV_files:
        df = pd.read_csv(file, dtype={"Type": "category", "Namespace": "category"}).rename(
            columns=lambda s: s.replace(" ", "_").lower()).set_index("model_id")
        this_valid_model_id = set(df.index) - (set(df.index) - set(valid_model_id))
        for model_id in this_valid_model_id:
            valid_models_json = df.loc[model_id]['model_json']
            tree = None
            try:
                # Load and convert json-based BPMN into Petri net
                net, initial_marking, final_marking = converter.convert_to_petri_net(
                    valid_models_json, False)

                tree = pm4py.convert_to_process_tree(net, initial_marking, final_marking)

            except:
                print("WARNING: Error during conversion from bpmn to Petri net.")
                failed += 1

            if tree is not None:
                flag = to_petri_helper(tree, model_id)

                if flag:
                    success += 1
                else:
                    failed += 1

            print('convert_BPMAI_jsons_to_petri:', success + failed, "jsons done. Succes: ", success, "failed: ",
                  failed)
            if (success + failed) % 50 == 0:
                gc.collect()


def generate_logs_from_petri_sers():
    pnet_ser_files = [f for f in os.listdir(petri_dir) if f.endswith(".pnser")]
    pnet_ser_files.sort()
    print(f"Start generating logs from petri_sers.")
    print("Total number of pnet files:", len(pnet_ser_files))
    success = 0
    done = 0

    for ser_file in pnet_ser_files:
        print('Started parsing:', ser_file)
        # ser_file = '8f45aff6468b4c2e985db4e694baf889.pnser'
        case_name = os.path.basename(ser_file).split('.')[0]
        filepath = os.path.join(petri_dir, ser_file)
        if os.path.getsize(filepath) > 0:
            net, initial_marking, final_marking = pickle.load(open(filepath, 'rb'))

            # from pm4py.visualization.petri_net import visualizer as pn_visualizer
            # gviz = pn_visualizer.apply(net, initial_marking, final_marking)
            # pn_visualizer.view(gviz)

            log = pna.playout_net(net, initial_marking, final_marking, playout_timeout)
            # log_no_loops = pna.create_log_without_loops(log)
            if len(log) > 0:
                xes_file = os.path.join(model_log_dir, case_name + ".xes")
                pm4py.write_xes(log, xes_file)

                # xes_file2 = os.path.join(target_dir_no_loops, case_name + ".xes")
                # pm4py.write_xes(log_no_loops, xes_file2)
                # print(f"Saved as model (log) {xes_file}")

                success += 1
            else:
                print('no valid trace in ' + ser_file)
        done += 1
        print(f"Number of Successfully converted models: {success} / {done}")
        if done % 25 == 0:
            gc.collect()
    print(f"Generate logs from petri_sers completed.")


def get_all_act_names():
    print('Start getting all activity names')
    xess = os.listdir(model_log_dir)
    act_names = set()

    for xes in tqdm(xess):
        log = pm4py.read_xes(os.path.join(model_log_dir, xes))

        for i in range(len(log)):
            trace = log[i]

            for j in range(len(trace)):
                event_name = trace[j]['concept:name']
                act_names.add(event_name)

    with open(act_name_path, 'w') as f:
        json.dump(list(act_names), f)


def gen_exclusive_anomalies(normal_variants, exclusives, base_name, model_log_path):
    exclusive_anomalies = []
    ith_a = 1
    for i, exclusive in enumerate(exclusives):
        net, initial_marking, final_marking = pickle.load(
            open(os.path.join(exclusive_anomalous_petri_dir, f'{base_name}_{i}.pnser'), 'rb'))
        simulated_log = pna.playout_net(net, initial_marking, final_marking, playout_timeout, 'basic')
        variants = set()
        for case in simulated_log:
            variant = []
            for event in case:
                variant.append(event['concept:name'])
            variants.add(tuple(variant))

        if len(variants)<2:
            continue

        random_variants = random.sample(variants, 2) # only produce two anomalous traces

        rand_ints = random.sample(range(len(exclusive)), 2)

        for j, variant in enumerate(random_variants):
            rand_int = rand_ints[j]

            events = []
            for act in variant:
                events.append({'name': act, 'timestamp': None})


            exclusive1 = set(exclusive[rand_int])

            nested_lists = exclusive[:rand_int] + exclusive[rand_int + 1:]
            exclusive2 = set([item for sublist in nested_lists for item in sublist])

            part1 = list(set(variant) & exclusive1)
            part2 = list(set(variant) & exclusive2)

            exclusive_anomaly = {'attributes': {"concept:name": f'ExclusiveAnomaly{ith_a}',
                                                'label': {'anomaly': 'Exclusive',
                                                          'attr': {'part1': part1, 'part2': part2}}}, 'events': events}
            exclusive_anomalies.append(exclusive_anomaly)
            ith_a += 1
    return exclusive_anomalies


def gen_artificial_anomalies(times=2):
    print('Start generating artificial anomalies')
    model_log_names = os.listdir(model_log_dir)

    anomaly_types = [
        SkipSequenceAnomaly(max_sequence_size=3),
        ReworkAnomaly(max_distance=5, max_sequence_size=3),
        EarlyAnomaly(max_distance=5, max_sequence_size=3),
        LateAnomaly(max_distance=5, max_sequence_size=3),
        InsertAnomaly(max_inserts=3),
    ]
    with open(act_name_path) as f:
        act_names = json.load(f)

    for anomaly in anomaly_types:  # make random inserted activity
        anomaly.activities = act_names
        anomaly.activities_len = len(act_names)

    for model_log_name in tqdm(model_log_names):
        # print(model_log_path)
        base_name = os.path.split(model_log_name)[1].split('.')[0]
        model_log_path = os.path.join(model_log_dir, model_log_name)

        anomaly_cases = []
        model_log = EventLog.from_xes(model_log_path)

        variants = set()
        for case in model_log:
            variant = []
            for event in case:
                variant.append(event.name)
            variants.add(tuple(variant))

        for case in model_log:
            NoneAnomaly().apply_to_case(case)

        for i in range(times):  # the injection times for each anomaly type
            for anomaly in anomaly_types:
                for case in model_log:
                    a_case = copy.deepcopy(case)
                    anomaly.apply_to_case(a_case)
                    anomaly_cases.append(a_case)

        model_log.cases += anomaly_cases

        exclusives_path = os.path.join(exclusive_dir, f'{base_name}.json')
        with open(exclusives_path) as f:
            exclusives = json.load(f)

        if len(exclusives) > 0:
            exclusive_anomalies = gen_exclusive_anomalies(variants, exclusives, base_name, model_log_path)
            result = model_log.json
            result['cases'] += exclusive_anomalies
            with open(os.path.join(anomalous_model_log_dir, base_name + '.json'), 'wt') as f:
                json.dump(result, f, sort_keys=True, indent=4, separators=(',', ': '))
        else:
            model_log.save_json(os.path.join(anomalous_model_log_dir, base_name + '.json'))

def write_file(data_list, path):
    # 写入文件
    with open(path, 'a') as file:
        for dictionary in data_list:
            file.write(json.dumps(dictionary))
            file.write("\n")  # 在字典之间插入空行


def extract_cause(label, trace):
    cause = ''
    if label['anomaly'] == 'SkipSequence':
        size = label['attr']['size']
        start = label['attr']['start']
        skipped_seq = [event['name'] for event in label['attr']['skipped']]
        if size == 1:
            cause = 'The activity \'{}\' is skipped before \'{}\'.'.format(skipped_seq[0], trace[start])
        elif size == 2:
            cause = 'The activities \'{}\' and \'{}\' are skipped before \'{}\'.'.format(skipped_seq[0], skipped_seq[1],
                                                                                         trace[start])
        else:
            cause = 'The activities \'{}\' and \'{}\' are skipped before \'{}\'.'.format('\',\''.join(skipped_seq[:-1]),
                                                                                         skipped_seq[-1], trace[start])

    if label['anomaly'] == 'Rework':
        size = label['attr']['size']
        start = label['attr']['start']
        reworked_seq = [event['name'] for event in label['attr']['inserted']]
        if size == 1:
            cause = 'The activity \'{}\' is reworked after \'{}\'.'.format(reworked_seq[0], trace[start - 1])
        elif size == 2:
            cause = 'The activities \'{}\' and \'{}\' are reworked after \'{}\.'.format(reworked_seq[0],
                                                                                         reworked_seq[1],
                                                                                         trace[start - 1])
        else:
            cause = 'The activities \'{}\' and \'{}\' are reworked after \'{}\'.'.format(
                '\',\''.join(reworked_seq[:-1]),
                reworked_seq[-1], trace[start - 1])

    if label['anomaly'] == 'Early':
        size = label['attr']['size']
        shift_from = label['attr']['shift_from']
        shift_to = label['attr']['shift_to']
        early_seq = trace[shift_to: shift_to + size]
        if size == 1:
            cause = 'The activity \'{}\' is executed too early, it should be executed after \'{}\'.'.format(
                early_seq[0], trace[shift_from - 1])
        elif size == 2:
            cause = 'The activities \'{}\' and \'{}\' are executed too early, they should be executed after \'{}\'.'.format(
                early_seq[0], early_seq[1],
                trace[shift_from - 1])
        else:
            cause = 'The activities \'{}\' and \'{}\' are executed too early, they should be executed after \'{}\'.'.format(
                '\',\''.join(early_seq[:-1]),
                early_seq[-1], trace[shift_from - 1])

    if label['anomaly'] == 'Late':
        size = label['attr']['size']
        shift_from = label['attr']['shift_from']
        shift_to = label['attr']['shift_to']
        early_seq = trace[shift_to: shift_to + size]
        if size == 1:
            cause = 'The activity \'{}\' is executed too late, it should be executed before \'{}\'.'.format(
                early_seq[0], trace[shift_from])
        elif size == 2:
            cause = 'The activities \'{}\' and \'{}\' are executed too late, they should be executed before \'{}\'.'.format(
                early_seq[0], early_seq[1],
                trace[shift_from])
        else:
            cause = 'The activities \'{}\' and \'{}\' are executed too late, they should be executed before \'{}\'.'.format(
                '\',\''.join(early_seq[:-1]),
                early_seq[-1], trace[shift_from])

    if label['anomaly'] == 'Insert':
        indices = label['attr']['indices']
        size = len(indices)
        wrong_acts = [trace[index] for index in indices]
        if size == 1:
            cause = 'The activity \'{}\' should not be executed.'.format(wrong_acts[0])
        else:
            cause = 'The activities \'{}\' and \'{}\' should not be executed.'.format('\',\''.join(wrong_acts[:-1]),
                                                                                      wrong_acts[-1])

    if label['anomaly'] == 'Exclusive':
        part1 = label['attr']['part1']
        part2 = label['attr']['part2']

        if len(part1)>1:
            act_1 = 'activities'
            str_1 = "{}' and '{}".format('\',\''.join(part1[:-1]), part1[-1])
            du = 'are'
        else:
            act_1 = 'activity'
            str_1 = part1[0]
            du = 'is'
        if len(part2)>1:
            act_2 = 'activities'
            str_2 = "{}' and '{}".format('\',\''.join(part2[:-1]), part2[-1])
        else:
            act_2 = 'activity'
            str_2 = part2[0]

        cause = "The {} '{}' {} mutually exclusive with the {} '{}', meaning they should not be executed within the same process instance.".format(act_1,str_1,du,act_2,str_2)


    return cause



def gen_train_test_spilt_data(times=2):
    '''
           use 1000 models for generating testing data and the rest of models for generating training data
    '''
    dataset_names = os.listdir(anomalous_model_log_dir)
    random.shuffle(dataset_names)

    dataset_names_train = dataset_names[:-1000]
    dataset_names_test = dataset_names[-1000:]

    print(len(dataset_names_train))
    print(len(dataset_names_test))


    with open(os.path.join(base_dir, 'train_dataset.jsonl'), 'w') as file:
        file.write('')
    data_list = []
    for ith, dataset_name in enumerate(tqdm(dataset_names_train)):  #gen train
        with open(os.path.join(anomalous_model_log_dir, dataset_name), "r") as json_file:
            data = json.load(json_file)
            normal_traces = set()  # normal trace is all at the begining of the file.
            for case in data['cases']:
                trace = []
                for event in case['events']:
                    trace.append(event['name'])
                label = case['attributes']['label']
                if label == 'normal':
                    this_data_dict = {}
                    this_data_dict['trace'] = '[' + ','.join(trace) + ']'
                    this_data_dict['label'] = label
                    this_data_dict['cause'] = ''
                    [data_list.append(this_data_dict) for _ in range(times * 5)]
                    normal_traces.add(tuple(trace))
                else:  # 加两条训练数据，一条带cause 一条不带cause
                    if tuple(trace) in normal_traces:
                        continue

                    this_data_dict = {}
                    this_data_dict['trace'] = '[' + ','.join(trace) + ']'
                    this_data_dict['label'] = 'anomalous'
                    # this_data_dict['raw_cause'] = label
                    this_data_dict['cause'] = extract_cause(label, trace)
                    data_list.append(this_data_dict)

                    this_data_dict = {}
                    this_data_dict['trace'] = '[' + ','.join(trace) + ']'
                    this_data_dict['label'] = 'anomalous'
                    # this_data_dict['raw_cause'] = label
                    this_data_dict['cause'] = ''
                    data_list.append(this_data_dict)
        if ith % 1000 == 0:
            write_file(data_list,os.path.join(base_dir, 'train_dataset.jsonl'))
            data_list = []
    if len(data_list) > 0:
        write_file(data_list,os.path.join(base_dir, 'train_dataset.jsonl'))



########################gen test
    with open(os.path.join(base_dir, 'test_dataset_1.jsonl'), 'w') as file:
        file.write('')
    with open(os.path.join(base_dir, 'test_dataset_cause_1.jsonl'), 'w') as file:
        file.write('')


    data_list = []
    data_cause_list = []
    for ith, dataset_name in enumerate(tqdm(dataset_names_test)): #gen test
        with open(os.path.join(anomalous_model_log_dir, dataset_name), "r") as json_file:
            data = json.load(json_file)
            normal_traces = set()  # normal trace is all at the begining of the file.
            for case in data['cases']:
                trace = []
                for event in case['events']:
                    trace.append(event['name'])
                label = case['attributes']['label']
                if label == 'normal':
                    if tuple(trace) in normal_traces:
                        continue

                    normal_traces.add(tuple(trace))
                    this_data_dict = {}
                    this_data_dict['trace'] = '[' + ','.join(trace) + ']'
                    this_data_dict['label'] = label
                    this_data_dict['raw_cause'] = ''
                    this_data_dict['cause'] = ''
                    data_list.append(this_data_dict)
                else:  # 加条训练数据，不带cause ,只为了评估异常检测结果（normal or anomalous）
                    if tuple(trace) in normal_traces or random.random() < 1 - 1 / (
                            5 * times):  # 让正确的轨迹不要错误地添加，并且有只有1/(5*times)的概率添加（平衡正负样本）。
                        continue

                    this_data_dict = {}
                    this_data_dict['trace'] = '[' + ','.join(trace) + ']'
                    this_data_dict['label'] = 'anomalous'
                    this_data_dict['raw_cause'] = label['anomaly']
                    this_data_dict['cause'] = ''
                    data_list.append(this_data_dict)

                    this_data_dict = {}
                    this_data_dict['trace'] = '[' + ','.join(trace) + ']'
                    this_data_dict['label'] = 'anomalous'
                    this_data_dict['raw_cause'] = label['anomaly']
                    this_data_dict['cause'] = extract_cause(label, trace)
                    data_cause_list.append(this_data_dict)

        if ith % 1000 == 0:
            write_file(data_list, os.path.join(base_dir, 'dataset/test_dataset_1.jsonl'))  # 只为了评估异常检测结果
            data_list = []
            write_file(data_cause_list, os.path.join(base_dir, 'dataset/test_dataset_cause_1.jsonl'))  # 只为了评估异常原因
            data_cause_list = []
    if len(data_list) > 0:
        write_file(data_list, os.path.join(base_dir, 'dataset/test_dataset_1.jsonl'))
        write_file(data_cause_list, os.path.join(base_dir, 'dataset/test_dataset_cause_1.jsonl'))


if __name__ == '__main__':
    convert_BPMAI_jsons_to_petri()  # convert BPMAI (https://zenodo.org/records/3758705) dataset into petri net
    convert_FBPM_BPMN_to_petri() # convert FBPM (http://fundamentals-of-bpm.org/process-model-collections/) dataset into petri net
    convert_SAP_jsons_to_petri() # convert SAP-SAM (https://zenodo.org/records/7012043) dataset into petri net
    generate_logs_from_petri_sers()
    get_all_act_names()
    gen_artificial_anomalies(2)
    gen_train_test_spilt_data(2)
