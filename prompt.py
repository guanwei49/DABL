preface1 = [
    'In the following business process trace, ',
    'In the subsequent business process trace, ',
    'In the forthcoming business process trace, ',
    'In the ensuing business process trace, ',
    'Below is a business process trace, ',
    'Below is a trace of a business process, ',
    'Here is a trace of a business process, ',
]

preface2 = [
    'each executed activity is separated by a comma:',
    'each performed activity is delimited by a comma:',
    'each activity that has been executed is delineated by a comma:',
    'each completed activity is demarcated by a comma:',
    'with each executed activity separated by a comma:',
    'with each completed activity separated by a comma:',
    'with each performed activity separated by a comma:'
]

ask_cause = [
    "What's the cause of the anomaly?",
    "What's the root cause of the anomaly?",
    "What's the root reason of the anomaly?",
    "What's the reason of the anomaly?",
    "What's the underlying cause of the anomaly?",
    "What's the reason behind this trace being anomalous?",
    "What's the reason behind this anomalous trace?",
    "What's the origin of the anomaly?",
    "What's the underlying reason for the anomaly?",
    "What's the primary factor behind the anomaly?",
    "What's the main explanation for the anomaly?",
    "What's the core issue leading to the anomaly?",
    "What causes this trace anomalous?",
    "What triggers the anomaly?",
    "What leads to the anomaly?",
    'What makes this trace anomalous?',
    "What causes this trace to deviate?",
    "What brings about this trace's anomaly?",
    "What accounts for this trace's anomaly?",
    "What attributes to the trace's anomaly?",
    "Why is this an anomalous trace?"
]


## generate prompt based on template ###
prompt_template = {
    "prompt_no_cause_q": "{p1}{p2} {trace}. Is this trace normal or anomalous?",

    "prompt_no_cause_all": "{p1}{p2} {trace}. Is this trace normal or anomalous? \\n The trace is {label}.</s>",

    "prompt_with_cause_all": "{p1}{p2} {trace}. Is this trace normal or anomalous? \\n The trace is {label}. \\n {ask_c} \\n {cause}</s>",

    "prompt_with_cause_q": "{p1}{p2} {trace}. Is this trace normal or anomalous? \\n The trace is {label}. \\n {ask_c}"
}
