import networkx as nx
from todoist_api_python.api import TodoistAPI, Project, Task, Comment
from argparse import ArgumentParser
from typing import Iterable, Union, Optional, Dict, List, Tuple
import pyvis


NEXT_ACTION_LABEL = "next-action"

def parse_args():
    ap = ArgumentParser("macher")
    ap.add_argument("-t", "--token", required=True)

    return ap.parse_args()


def get_parent(obj: Union[Project, Task]):
    if isinstance(obj, Project) and obj.parent_id != None:
        return obj.parent_id
    elif isinstance(obj, Task) and obj.project_id != None:
        return obj.project_id
    else:
        return None


def get_label(obj: Union[Project, Task]):
    if (result := getattr(obj, "name", None)) is not None:
        return result
    elif (result := getattr(obj, "content", None)) is not None:
        return result
    else:
        return None


def get_size(obj: Union[Project, Task, Comment]):
    if isinstance(obj, Project):
        return 30
    elif isinstance(obj, Task):
        return 20
    else:
        raise ValueError


def get_color(obj: Union[Project, Task, Comment]):
    if isinstance(obj, Project):
        return "blue"
    elif isinstance(obj, Task):
        return "red"
    else:
        raise ValueError


from enum import Enum, auto


class ProjectStructure(Enum):
    SERIAL        = auto()
    PARALLEL      = auto()
    COLLECTION    = auto()

    STRUCTURELESS = auto()


REFERENCE_KEY = "ref"


# def TodoistNetwork:
#     def __init__(self, api):
#         self.projects = {p.id: p for p in api.get_projects()}
#         self.tasks = {t.id: t for t in api.get_tasks()}
#         self.comments = dict()  # api.get_comments()

#         nx_graph = todoist_network(self.projects.values(), self.tasks.values(), self.comments.values())

#     def get_by_id(self, obj_id):
#         results = []

#         results.append(self.projects.get(obj_id, None))
#         results.append(self.tasks.get(obj_id, None))
#         results.append(self.comments.get(obj_id, None))

#         real_results = list(filter(iterable=results))

#         if len(real_results) > 1:
#             raise RuntimeError
#         elif len(real_results) == 0:
#             raise KeyError
#         else:
#             return real_results[0]


def get_obj_by_id(G: nx.DiGraph, node_id: str) -> Union[Project, Task]:
    obj = G.nodes[node_id][REFERENCE_KEY]
    return obj


def is_leaf(G: nx.DiGraph, node_id: str) -> bool:
    return len(list(G.successors(node_id))) == 0


def is_leaf_project(G: nx.DiGraph, node_id: str):
    obj = get_obj_by_id(G, node_id)

    if not isinstance(obj, Project):
        return False

    for child_id in G.successors(node_id):
        child_obj = get_obj_by_id(G, child_id)

        if isinstance(child_obj, Project):
            return False

    return True


# import numpy as np

structure_lut: Dict[str, ProjectStructure] = {
    "+": ProjectStructure.COLLECTION,
    "=": ProjectStructure.PARALLEL,
    ">": ProjectStructure.SERIAL,
}


def ancestors(G, node_id):
    all_ancestors = set()
    parent_ids = G.predecessors(node_id)

    for parent_id in parent_ids:
        if parent_id not in all_ancestors:
            all_ancestors.update(ancestors(G, parent_id))
            all_ancestors.add(parent_id)

    return all_ancestors


def is_exempt(G: nx.DiGraph, node_id) -> bool:
    ancestor_ids = ancestors(G, node_id)
    ancestor_objs = [get_obj_by_id(G, parent_id) for parent_id in ancestor_ids]

    for ancestor_obj in ancestor_objs:
        if isinstance(ancestor_obj, Project) and ancestor_obj.name[-1] == "°":
            return True
    return False


def get_structure_by_project(project: Project) -> ProjectStructure:
    id_char = project.name[-1]

    if (structure := structure_lut.get(id_char, None)) is not None:
        return structure
    else:
        return ProjectStructure.STRUCTURELESS

import numpy as np

def get_next_actions(G: nx.DiGraph, root_id: str) -> List[str]:
    ref_obj = G.nodes[root_id][REFERENCE_KEY]

    next_ids: List[str] = []
    child_objects = [G.nodes[cid][REFERENCE_KEY] for cid in G.successors(root_id)]

    if is_exempt(G, root_id):
        return []
    elif is_leaf(G, root_id):
        if isinstance(ref_obj, Task):
            return [root_id]
        else:
            return []
    elif G.nodes[root_id]["structure"] == ProjectStructure.PARALLEL:
        for child_obj in child_objects:
            next_ids.extend(get_next_actions(G, child_obj.id))
    elif G.nodes[root_id]["structure"] == ProjectStructure.SERIAL:
        order_id_lut = {get_order(obj): obj.id for obj in child_objects}
        first = np.min(list(order_id_lut.keys()))
        next_ids.extend(get_next_actions(G, order_id_lut[first]))
    else:
        pass

    return next_ids

import re

def get_order(obj: Union[Task, Project]) -> int:
    text = ""
    order = 0
    if isinstance(obj, Project):
        text = obj.name
    elif isinstance(obj, Task):
        text = obj.content
    
    match = re.match(r"\((\d+)\).*", text)
    
    if match:
        order = int(match.group(1))
    else:
        order = obj.order
    
    return order

def get_structure(obj: Union[Project, Task]) -> ProjectStructure:
    if isinstance(obj, Task):
        labels = obj.labels

        the_structure = None
        for structure in [
            ProjectStructure.COLLECTION,
            ProjectStructure.PARALLEL,
            ProjectStructure.SERIAL,
        ]:
            if structure in labels:
                the_structure = structure

        if the_structure is None:
            the_structure = ProjectStructure.STRUCTURELESS
    elif isinstance(obj, Project):
        the_structure = get_structure_by_project(obj)
    else:
        raise RuntimeError

    return the_structure


def todoist_network(projects: Iterable[Project], tasks: Iterable[Task] = []):
    G = nx.DiGraph()

    for obj in [*projects, *tasks]:
        labels = getattr(obj, "labels", [])

        the_structure = get_structure(obj)

        G.add_node(
            obj.id,
            label=get_label(obj),
            title=the_structure,
            size=get_size(obj),
            color=get_color(obj),
            structure=the_structure,
            ref=obj,
        )

        parent_id = get_parent(obj)

        if parent_id is not None:
            G.add_edge(parent_id, obj.id)

    return G


def clean_graph(G, mark_nexts):
    g2 = nx.DiGraph()

    for node_id, data in G.nodes(data=True):
        data2 = data.copy()
        del data2[REFERENCE_KEY]

        if node_id in mark_nexts:
            data2["color"] = "green"

        g2.add_node(node_id, **data2)

    g2.add_edges_from(G.edges(data=True))

    return g2

from datetime import datetime as dt


def main():
    args = parse_args()
    api = TodoistAPI(args.token)

    projects = api.get_projects()
    tasks = api.get_tasks()

    nx_graph = todoist_network(projects, tasks)

    leaf_project_ids = [
        node_id for node_id in nx_graph.nodes() if is_leaf_project(nx_graph, node_id)
    ]

    all_nexts = set()
    for project_id in leaf_project_ids:
        nexts = get_next_actions(nx_graph, project_id)
        print(project_id, nexts)
        all_nexts.update(nexts)
        
    for task in tasks:
        labels = task.labels
        order = get_order(task)
        
        print(task)
        
        if task.id in all_nexts:
            if NEXT_ACTION_LABEL not in task.labels:
                labels = task.labels + [NEXT_ACTION_LABEL]
        else:
            if NEXT_ACTION_LABEL in task.labels:
                labels = list(set(task.labels).difference(set([NEXT_ACTION_LABEL])))
                
        if task.labels != labels or task.order != order:
            update_result = hard_update_task(api, task, labels=labels, order=order)
            print(f"Update: {task} => {update_result}")
                
                
    for task in tasks:
        if task.content == "Last Macher Run":
            api.update_task(task.id, description=dt.now().strftime("%Y-%m-%d, %H:%M:%S"))

    # nt = pyvis.network.Network(height="420px", notebook=True, directed=True)
    # # populates the nodes and edges data structures
    # pgraph = clean_graph(nx_graph, mark_nexts=all_nexts)
    # nt.from_nx(pgraph)
    # nt.toggle_physics(False)
    # nt.show_buttons(filter_=["physics"])
    # nt.show("nx.html")


def hard_update_task(api: TodoistAPI, task: Task, labels=None, order=None):
    new_order = order or task.order
    new_labels = labels or task.labels
    
    new_task = api.add_task(
        content=task.content,
        description=task.description,
        project_id=task.project_id,
        section_id=task.section_id,
        parent_id=task.parent_id,
        order=new_order,
        labels=new_labels,
        priority=task.priority,
        due_date=task.due.date if task.due else None,
        assignee_id=task.assignee_id,
    )
    api.delete_task(task.id)
    
    return task

if __name__ == "__main__":
    main()
