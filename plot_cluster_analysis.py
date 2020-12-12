import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go


def generate_cluster_graph(visual_info_file_path):
    dataset = pd.read_csv(visual_info_file_path)
    iterations = dataset["iteration"].drop_duplicates().tolist()

    component1_max_val = max(dataset["component1"].tolist())
    component1_min_val = min(dataset["component1"].tolist())

    component2_max_val = max(dataset["component2"].tolist())
    component2_min_val = min(dataset["component2"].tolist())

    clusters = dataset["cluster_id"].drop_duplicates().tolist()

    # make figure
    fig_dict = {
        "data": [],
        "layout": {},
        "frames": []
    }

    # fill in most of layout
    fig_dict["layout"]["xaxis"] = {"range": [component1_min_val - 30, component1_max_val + 30],
                                   "title": "PCA component1"}
    fig_dict["layout"]["yaxis"] = {"title": "PCA component2",
                                   "range": [component2_min_val - 30, component2_max_val + 30]}
    fig_dict["layout"]["hovermode"] = "closest"
    fig_dict["layout"]["updatemenus"] = [
        {
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": 500, "redraw": False},
                                    "fromcurrent": True, "transition": {"duration": 300,
                                                                        "easing": "quadratic-in-out"}}],
                    "label": "Play",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                      "mode": "immediate",
                                      "transition": {"duration": 0}}],
                    "label": "Pause",
                    "method": "animate"
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }
    ]

    sliders_dict = {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 20},
            "prefix": "Iteration:",
            "visible": True,
            "xanchor": "right"
        },
        "transition": {"duration": 300, "easing": "cubic-in-out"},
        "pad": {"b": 10, "t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": []
    }

    # make data
    iteration = 0
    for cluster in clusters:
        dataset_by_iteration = dataset[dataset["iteration"] == iteration]
        dataset_by_iter_and_cluster = dataset_by_iteration[
            dataset_by_iteration["cluster_id"] == cluster]

        data_dict = {
            "x": list(dataset_by_iter_and_cluster["component1"]),
            "y": list(dataset_by_iter_and_cluster["component2"]),
            "mode": "markers",
            "text": list(dataset_by_iter_and_cluster["cluster_id"]),
            "marker": {
                "sizemode": "area",
                "sizeref": 10000,
                "size": list(dataset_by_iter_and_cluster["cluster_size"])
            },
            "name": cluster
        }
        fig_dict["data"].append(data_dict)

    # make frames
    for iteration in iterations:
        frame = {"data": [], "name": str(iteration)}
        for cluster in clusters:
            dataset_by_iteration = dataset[dataset["iteration"] == iteration]
            dataset_by_iter_and_cluster = dataset_by_iteration[
                dataset_by_iteration["cluster_id"] == cluster]

            data_dict = {
                "x": list(dataset_by_iter_and_cluster["component1"]),
                "y": list(dataset_by_iter_and_cluster["component2"]),
                "mode": "markers",
                "text": list(dataset_by_iter_and_cluster["cluster_id"]),
                "marker": {
                    "sizemode": "area",
                    "sizeref": 10,
                    "size": list(dataset_by_iter_and_cluster["cluster_size"])
                },
                "name": cluster
            }
            frame["data"].append(data_dict)

        fig_dict["frames"].append(frame)
        slider_step = {"args": [
            [iteration],
            {"frame": {"duration": 300, "redraw": False},
             "mode": "immediate",
             "transition": {"duration": 300}}
        ],
            "label": iteration,
            "method": "animate"}
        sliders_dict["steps"].append(slider_step)

    fig_dict["layout"]["sliders"] = [sliders_dict]

    fig = go.Figure(fig_dict)

    plotly.offline.plot(fig, filename="cluster_analysis.html", auto_open=False)


def generate_loss_graph(analysis_info_file_path):
    df = pd.read_csv(analysis_info_file_path)
    df["cluster_id"] = df["cluster_id"].apply(lambda x: "cluster_{0}".format(x))
    df = df.sort_values(by="iteration")

    fig = px.bar(df, x="cluster_id", y="loss", color="cluster_id",
                 animation_frame="iteration", animation_group="cluster_id", range_y=[0, 1000],
                 title="Sum of Square Error")

    plotly.offline.plot(fig, filename="cluster_sse.html", auto_open=False)


def generate_cluster_purity_graph(analysis_info_file_path):
    df = pd.read_csv(analysis_info_file_path)
    df["cluster_id"] = df["cluster_id"].apply(lambda x: "cluster_{0}".format(x))
    df = df.sort_values(by="iteration")

    fig = px.bar(df, x="cluster_id", y="cluster_purity", color="cluster_id",
                 animation_frame="iteration", animation_group="cluster_id", range_y=[0, 0.2], title="Cluster Purity")

    plotly.offline.plot(fig, filename="cluster_purity.html", auto_open=False)


if __name__ == '__main__':
    generate_cluster_graph("cluster_visual.csv")
    generate_loss_graph("cluster_analysis.csv")
    generate_cluster_purity_graph("cluster_analysis.csv")
