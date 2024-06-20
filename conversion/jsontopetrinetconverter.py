from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils import petri_utils as utils

import conversion.bpmnjsonanalyzer as bpmn_analyzer


class JsonToPetriNetConverter:

    def __init__(self):
        self.key_index = 0

    def convert_to_petri_net(self,file,is_file_path=True):

        follows, labels, tasks = bpmn_analyzer.loadJSON(file,is_file_path)
        net = PetriNet("net")
        sources = set()
        sinks = set()
        elements = {}
        gateways_input = {}
        gateways_output = {}
        implicit_joins = set()
        self.key_index = 0
        irrelevant_shapes = ("SequenceFlow", "MessageFlow", "DataObject", "Pool", "Lane", "TextAnnotation", "Association_Undirected", "Association_Bidirectional", "Association_Unidirectional","Group", "CollapsedPool", "ITSystem", "DataStore")

        for s in follows.keys():

            # Handle events
            if bpmn_analyzer._get_type(s, labels, irrelevant_shapes) == "Event":

                # Special case: attached events
                if labels[s].startswith("Intermediate") & len([x for x in bpmn_analyzer._get_preset(labels, follows, s) if bpmn_analyzer._is_relevant(x,labels,irrelevant_shapes)]) > 0:
                    origin = bpmn_analyzer._get_preset(labels, follows, s).pop()
                    if s in follows[origin]:
                        labels[s] = "AttachedEvent"
                p = PetriNet.Place(s)
                net.places.add(p)
                elements[s] = p
                if len(bpmn_analyzer._get_postset(labels, follows, s)) == 0:
                    sinks.add(s)
                if len(bpmn_analyzer._get_preset(labels, follows, s)) == 0:
                    sources.add(s)

            # Handle gateways
            elif bpmn_analyzer._get_type(s, labels, irrelevant_shapes) == "Gateway":
                t = PetriNet.Transition(s, labels[s])
                net.transitions.add(t)
                elements[s] = t

            # Handle tasks
            elif not labels[s].startswith(irrelevant_shapes):
                if labels[s].startswith("CollapsedSubprocess"):
                    labels[s] = labels[s][21:-1]
                t = PetriNet.Transition(s, labels[s])
                net.transitions.add(t)
                elements[s] = t

                # Implicit joins
                if len([x for x in bpmn_analyzer._get_preset(labels, follows, s) if bpmn_analyzer._is_relevant(x,labels,irrelevant_shapes)]) > 1:
                    implicit_joins.add(s)

                # Note that implicit splits requires no further handling since
                # they are  AND-splits and will be converted as such

        # Establish flow relations (may require generating additional places and transitions)
        for s in follows.keys():
            #print(f"FlOW: {labels[s]} ({s}) {[labels[x] for x in follows[s]]} ({follows[s]})")

            # Only check relevant shapes
            if bpmn_analyzer._is_relevant(s, labels, irrelevant_shapes):

                # Get postset of considered element "s"
                postset = [x for x in bpmn_analyzer._get_postset(labels, follows, s) if bpmn_analyzer._is_relevant(x,labels,irrelevant_shapes)]

                #++++++++++++++++++++++++++++++
                # Source = event
                # ++++++++++++++++++++++++++++++

                if bpmn_analyzer._get_type(s, labels, irrelevant_shapes) == "Event":
                    for elem in postset:

                        # Source = event & target = event
                        if bpmn_analyzer._get_type(elem, labels, irrelevant_shapes) == "Event":
                            t = self._get_new_transition(net,elements,labels)
                            utils.add_arc_from_to(elements[s], t, net)
                            utils.add_arc_from_to(t, elements[elem], net)

                        # Source = event & target = task, gateway or other
                        else:
                            utils.add_arc_from_to(elements[s], elements[elem], net)

                # ++++++++++++++++++++++++++++++
                # Source = task
                # ++++++++++++++++++++++++++++++
                if bpmn_analyzer._get_type(s, labels, irrelevant_shapes) == "Task":
                    for elem in postset:

                        # Source = task & target = gateway
                        if bpmn_analyzer._get_type(elem, labels, irrelevant_shapes) == "Gateway":

                            # Source = task & target = parallel gateway
                            if labels[elem].startswith("Parallel"):
                                p = self._get_new_place(net,elements,labels)
                                utils.add_arc_from_to(elements[s], p, net)
                                utils.add_arc_from_to(p, elements[elem], net)

                            # Source = task & target = choice gateway (requires creating a shared place)
                            if bpmn_analyzer._is_choice(elem,labels):

                                # If a shared place already exists, we use that one
                                if elem in gateways_input:
                                    p = gateways_input[elem]
                                    utils.add_arc_from_to(elements[s], p, net)

                                # Otherwise, we create a shared place
                                else:
                                    p = self._get_new_place(net,elements,labels)
                                    gateways_input[elem] = p
                                    utils.add_arc_from_to(elements[s], p, net)
                                    utils.add_arc_from_to(p, elements[elem], net)

                        # Source = task & target = task
                        if bpmn_analyzer._get_type(elem, labels, irrelevant_shapes) == "Task":
                            p = self._get_new_place(net,elements,labels)
                            utils.add_arc_from_to(elements[s], p, net)
                            utils.add_arc_from_to(p, elements[elem], net)

                        # Source = task & target = event
                        if bpmn_analyzer._get_type(elem, labels, irrelevant_shapes) == "Event":
                            utils.add_arc_from_to(elements[s], elements[elem], net)

                # ++++++++++++++++++++++++++++++
                # Source = gateway
                # ++++++++++++++++++++++++++++++
                if bpmn_analyzer._get_type(s, labels, irrelevant_shapes) == "Gateway":
                    for elem in postset:

                        # Source = parallel gateway (create place for each outgoing element)
                        if labels[s].startswith("Parallel"):

                            # Source = parallel gateway & target = event
                            if bpmn_analyzer._get_type(elem, labels, irrelevant_shapes) == "Event":
                                utils.add_arc_from_to(elements[s], elements[elem], net)

                            # Source = parallel gateway & target = task, gateway or other
                            else:
                                p = self._get_new_place(net,elements,labels)
                                gateways_output[s] = p
                                utils.add_arc_from_to(elements[s], p, net)
                                utils.add_arc_from_to(p, elements[elem], net)

                        # Source: choice gateway (create shared place)
                        if bpmn_analyzer._is_choice(s,labels):

                            # Source = choice gateway & target = event
                            if bpmn_analyzer._get_type(elem, labels, irrelevant_shapes) == "Event":

                                # If a shared place is already available, we use it
                                if s in gateways_output:
                                    p = gateways_output[s]
                                    t = self._get_new_transition(net,elements,labels)
                                    utils.add_arc_from_to(p, t, net)
                                    utils.add_arc_from_to(t, elements[elem], net)

                                # Otherwise, we create it
                                else:
                                    p = self._get_new_place(net,elements,labels)
                                    gateways_output[s] = p
                                    t = self._get_new_transition(net,elements,labels)
                                    utils.add_arc_from_to(elements[s], p, net)
                                    utils.add_arc_from_to(p, t, net)
                                    utils.add_arc_from_to(t, elements[elem], net)

                            # Source = choice gateway & target = gateway
                            elif bpmn_analyzer._get_type(elem, labels, irrelevant_shapes) == "Gateway":

                                if s in gateways_output:
                                    output_p = gateways_output[s]
                                else:
                                    output_p = self._get_new_place(net,elements,labels)
                                    gateways_output[s] = output_p
                                    utils.add_arc_from_to(elements[s], output_p, net)

                                # Source = choice gateway & target = choice gateway
                                if bpmn_analyzer._is_choice(elem, labels):

                                    # If there is already an input place, we will use it
                                    if elem in gateways_input:
                                        input_p = gateways_input[elem]

                                    # If not, we check whether there is another event in the preset or create one
                                    else:
                                        input_p = None
                                        preset = self._get_net_preset(elements[elem])
                                        if len(preset) > 0:
                                            input_p = preset.pop()

                                        # If not, create one
                                        if input_p == None:
                                            input_p = self._get_new_place(net, elements, labels)
                                            gateways_input[elem] = input_p
                                            utils.add_arc_from_to(input_p, elements[elem], net)

                                    t = self._get_new_transition(net, elements, labels)
                                    utils.add_arc_from_to(output_p, t, net)
                                    utils.add_arc_from_to(t, elements[str(input_p)], net)

                                # Source = choice gateway & target = parallel gateway
                                else:
                                    if s in gateways_output:
                                        p_out = gateways_output[s]
                                    else:
                                        p_out = self._get_new_place(net, elements, labels)
                                        gateways_output[s] = p_out
                                    p_in = self._get_new_place(net, elements, labels)
                                    t = self._get_new_transition(net, elements, labels)
                                    utils.add_arc_from_to(p_out, t, net)
                                    utils.add_arc_from_to(t, p_in, net)
                                    utils.add_arc_from_to(p_in, elements[elem], net)

                            # Source = gateway & target = task or other
                            else:
                                if s in gateways_output:
                                    p = gateways_output[s]
                                    utils.add_arc_from_to(p, elements[elem], net)
                                else:
                                    p = self._get_new_place(net,elements,labels)
                                    gateways_output[s] = p
                                    utils.add_arc_from_to(elements[s], p, net)
                                    utils.add_arc_from_to(p, elements[elem], net)
        # Correct implicit joins
        for elem in implicit_joins:
            p = self._get_new_place(net, elements, labels)
            pn_elem = elements[elem]
            pn_elem_sources = [arc.source for arc in pn_elem.in_arcs]
            for a in pn_elem.in_arcs:
                net.arcs.remove(a)
            for s in pn_elem_sources:
                t = self._get_new_transition(net, elements, labels)
                utils.add_arc_from_to(s,t, net)
                utils.add_arc_from_to(t, p, net)
            utils.add_arc_from_to(p, pn_elem, net)

        # Add missing start / end event
        for s in follows.keys():
            if bpmn_analyzer._get_type(s,labels, irrelevant_shapes) == "Task":
                if len(bpmn_analyzer._get_preset(labels, follows,s)) == 0:
                    p = self._get_new_place(net, elements, labels)
                    utils.add_arc_from_to(p, elements[s], net)
                    sources.add(p)
                if len(bpmn_analyzer._get_postset(labels, follows,s)) == 0:
                    p = self._get_new_place(net, elements, labels)
                    utils.add_arc_from_to(elements[s], p, net)
                    sinks.add(p)

        # Multiple start events (Assumption: they are exclusive to each other)
        initial_marking = Marking()
        if len(sources) > 1:
            p = self._get_new_place(net, elements, labels)
            for elem in sources:
                t = self._get_new_transition(net, elements, labels)
                utils.add_arc_from_to(t, elements[str(elem)], net)
                utils.add_arc_from_to(p, t, net)
            initial_marking[p] = 1
        else:
            p = sources.pop()
            initial_marking[elements[str(p)]] = 1

        # Multiple end events (Assumption: they are exclusive to each other)
        final_marking = Marking()
        if len(sinks) > 1:
            p = self._get_new_place(net, elements, labels)
            for elem in sinks:
                t = self._get_new_transition(net, elements, labels)
                utils.add_arc_from_to(elements[str(elem)], t, net)
                utils.add_arc_from_to(t, p, net)
            final_marking[p] = 1
        else:
            p = sinks.pop()
            final_marking[elements[str(p)]] = 1

        # Handle attached events
        for s in follows.keys():
            if labels[s] == "AttachedEvent":
                origin = elements[bpmn_analyzer._get_preset(labels, follows, s).pop()]

                origin_output = set()
                for a in origin.out_arcs:
                    if (str(a.target) != str(s)):
                        origin_output.add(a.target)

                # If the set origin_output = 0, the event was not properly attached in the modeling editor
                if len(origin_output) > 0:
                    attached_event_place = elements[s]
                    split_place = elements[str(origin_output.pop())]
                    for a in attached_event_place.in_arcs:
                        net.arcs.remove(a)
                    t = self._get_new_transition(net, elements, labels)
                    utils.add_arc_from_to(split_place,t, net)
                    utils.add_arc_from_to(t, attached_event_place, net)

        # "Silence" gateway transitions by removing label
        gateway_labels = {"Exclusive_Databased_Gateway", "Inclusive_Gateway", "Eventbased_Gateway", "Parallel_Gateway"}
        for t in net.transitions:
            if t.label in gateway_labels:
                t.label = None

        return self._get_clean_copy(net, initial_marking, final_marking)

    def _get_net_preset(self, pn_elem):
        preset = set()
        for a in pn_elem.in_arcs:
           preset.add(a.source)
        return preset

    # Creates new place
    def _get_new_place(self, net, elements, labels):
        p = PetriNet.Place(self._get_new_key_index())
        net.places.add(p)
        elements[str(self._get_key_index())] = p
        labels[str(self._get_key_index())] = f"New place {self._get_key_index()}"
        return p

    # Creates new tau transition
    def _get_new_transition(self, net, elements,labels):
        t = PetriNet.Transition(self._get_new_key_index())
        net.transitions.add(t)
        elements[str(self._get_key_index())] = t
        labels[str(self._get_key_index())] = f"Task {str(self._get_key_index())}"
        return t

    # Generates new index for additional elements
    def _get_new_key_index(self):
        self.key_index += 1
        return str(self.key_index)

    # Returns current index (used for additional elements)
    def _get_key_index(self):
        return self.key_index

    def _get_clean_copy(self, net, initial_marking, final_marking):
        place_map = {}
        transition_map = {}
        net_copy = PetriNet("net copy")
        index = 1
        for p in net.places:
            p_new = PetriNet.Place(str(index))
            index += 1
            net_copy.places.add(p_new)
            place_map[p] = p_new
        for t in net.transitions:
            t_new = PetriNet.Transition(str(index), t.label)
            index += 1
            net_copy.transitions.add(t_new)
            transition_map[t] = t_new
        for a in net.arcs:
            if a.source in place_map.keys():
                utils.add_arc_from_to(place_map[a.source], transition_map[a.target], net_copy)
            else:
                utils.add_arc_from_to(transition_map[a.source], place_map[a.target], net_copy)

        im = Marking()
        for m in initial_marking.keys():
            im[place_map[m]] = initial_marking[m]
        fm = Marking()
        for m in final_marking.keys():
            fm[place_map[m]] = final_marking[m]
        return net_copy, im, fm



