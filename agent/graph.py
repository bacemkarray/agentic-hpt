from langgraph.graph import StateGraph, START, END
from agent import nodes, states

graph = (
    StateGraph(states.TuningState)
    # Nodes
    .add_node("define_parameters", nodes.initialize_params)
    .add_node("start_workers", nodes.start_workers)
    .add_node("tune_max_depth", nodes.make_worker("max_depth"))
    .add_node("tune_learning_rate", nodes.make_worker("learning_rate"))
    .add_node("tune_n_estimators",nodes. make_worker("n_estimators"))
    .add_node("tune_subsample", nodes.make_worker("subsample"))
    .add_node("coordinator", nodes.coordinator)
    .add_node("finalize", nodes.finalize)
    .add_node("wait", nodes.wait)

    # Edges
    .add_edge(START, "define_parameters")
    .add_edge("define_parameters", "start_workers")
    .add_edge("start_workers", "tune_max_depth")
    .add_edge("start_workers", "tune_learning_rate")
    .add_edge("start_workers", "tune_n_estimators")
    .add_edge("start_workers", "tune_subsample")
    .add_edge("tune_max_depth", "coordinator")
    .add_edge("tune_learning_rate", "coordinator")
    .add_edge("tune_n_estimators", "coordinator")
    .add_edge("tune_subsample", "coordinator")
    # .add_edge("coordinator", "wait")
    .add_edge("wait", "coordinator")
    .add_edge("finalize", END)
    .compile(name="Parallel HP Tuning Graph")
)


initial_input = {
    "max_depth": 6,
    "learning_rate": 0.1,
    "n_estimators": 100,
    "subsample": 1.0,
}

result = graph.invoke(initial_input)