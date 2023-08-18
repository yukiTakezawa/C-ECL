import random
import json
import argparse    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PowerECL')
    parser.add_argument('n_class', type=int)
    args = parser.parse_args()

    random.seed(0)
    n_node = 8
    n_class = args.n_class
    node_label = []
    for i in range(n_node):
        node_label.append(random.sample(range(10), n_class))

    # the list of nodes that have class i.
    label_node = []
    for i in range(10):
        label_node.append([])
        for node in range(n_node):
            if i in node_label[node]:
                label_node[-1].append(node)

    # check all label is contained.
    for i in range(10):
        if len(label_node[i]) == 0:
            raise ValueError(f"label {i} is not contaied!")
        else:
            print(f"label {i} is contained in {len(label_node[i])} nodes.")

    data = {"node0": {"adj" : [1, 7], "cuda" : "cuda:0", "n_class": node_label[0]},
            "node1": {"adj" : [0, 2], "cuda" : "cuda:0", "n_class": node_label[1]},
            "node2": {"adj" : [1, 3], "cuda" : "cuda:1", "n_class": node_label[2]},
            "node3": {"adj" : [2, 4], "cuda" : "cuda:1", "n_class": node_label[3]},
            "node4": {"adj" : [3, 5], "cuda" : "cuda:2", "n_class": node_label[4]},
            "node5": {"adj" : [4, 6], "cuda" : "cuda:2", "n_class": node_label[5]},
            "node6": {"adj" : [5, 7], "cuda" : "cuda:3", "n_class": node_label[6]},
            "node7": {"adj" : [6, 0], "cuda" : "cuda:3", "n_class": node_label[7]}}

    json.dump(data, open("ring_class" + str(n_class) + ".json", "w"))
