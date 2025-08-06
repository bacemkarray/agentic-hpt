from langgraph.graph import StateGraph, START, END
from agent import nodes, states

graph = (
    StateGraph(states.TuningState)
    # Nodes
    .add_node("define_parameters", nodes.initialize_params)
    .add_node("start_workers", nodes.start_workers)
    .add_node("tune_num_layers", nodes.make_worker("num_layers"))
    .add_node("tune_learning_rate", nodes.make_worker("learning_rate"))
    .add_node("tune_hidden_dim",nodes. make_worker("hidden_dim"))
    .add_node("tune_dropout", nodes.make_worker("dropout"))
    .add_node("coordinator", nodes.coordinator)
    .add_node("finalize", nodes.finalize)
    .add_node("wait", nodes.wait)

    # Edges
    .add_edge(START, "define_parameters")
    .add_edge("define_parameters", "start_workers")
    .add_edge("start_workers", "tune_num_layers")
    .add_edge("start_workers", "tune_learning_rate")
    .add_edge("start_workers", "tune_hidden_dim")
    .add_edge("start_workers", "tune_dropout")
    .add_edge("tune_num_layers", "coordinator")
    .add_edge("tune_learning_rate", "coordinator")
    .add_edge("tune_hidden_dim", "coordinator")
    .add_edge("tune_dropout", "coordinator")
    .add_edge("wait", "coordinator")
    .add_edge("finalize", END)
    .compile(name="Parallel HP Tuning Graph")
)

# Uncomment and edit this (as needed) if you wish to run the graph here.
# initial_input = {
#     "num_layers": 6,
#     "learning_rate": 0.1,
#     "hidden_dim": 100,
#     "dropout": 1.0,
# }

# result = graph.invoke(initial_input)