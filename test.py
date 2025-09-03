import torch
from TransformerMTGP.model.model import MyNN, SharedEmbeddings
from TransformerMTGP.src.classes.individual import Individual
from IPython.core.display import display
import numpy as np

import bertviz

if __name__ == "__main__":
    loaded_checkpoint = torch.load(
        "/Users/maoshanshi/Workspace/code/test/data/046cd36bd4f_single_model_0.8/checkpoint_40.pth"
        # "data/ab7e39e061a_single_model_nesi_0.5/checkpoint_30.pth"
    )
    shared_emb = SharedEmbeddings()
    shared_emb.load_state_dict(loaded_checkpoint["embedding_state_dict"])
    transformer_model = MyNN(64, 1024, 1, 8, 3, shared_emb)
    transformer_model.load_state_dict(loaded_checkpoint["model_state_dict"])
    transformer_model.eval()
    i = Individual(
        "add('OWT', multiply(multiply(add('PT', multiply(multiply(add('PT', 'MWT'), multiply('PT', 'WIQ')), multiply(multiply(add('PT', 'OWT'), multiply('PT', 'PT')), 'WIQ'))), 'PT'), multiply('PT', 'WIQ')))",
        "maximum('PT', maximum('SLACK', subtract('SLACK', subtract('MWT', subtract('PT', subtract('NIQ', maximum('PT', 'NIQ')))))))",
    )
    combined_x = torch.cat([i.route_data, i.sequence_data], dim=0)
    indices = torch.nonzero(combined_x[:, 1:] == 1, as_tuple=True)[1] + 1
    segement_ids = torch.tensor(
        [1] * i.route_data.size(0) + [2] * i.sequence_data.size(0), dtype=torch.long
    )

    output, score_vector, attention_weights = transformer_model(
        indices, segement_ids, is_batch=False
    )
    score_vector = score_vector.clone().detach().cpu().numpy()

    mask1 = segement_ids == 1
    mask2 = segement_ids == 2
    l_score_vector = score_vector[mask1.clone().detach().cpu().numpy()]
    r_score_vector = score_vector[mask2.clone().detach().cpu().numpy()]

    l_min = np.argmin(l_score_vector)
    l_max = np.argmax(l_score_vector)
    r_min = np.argmin(r_score_vector)
    r_max = np.argmax(r_score_vector)

    html_obj = bertviz.model_view(
        attention_weights,
        [
            "maximum",
            "PT",
            "maximum",
            "SLACK",
            "subtract",
            "SLACK",
            "subtract",
            "MWT",
            "subtract",
            "PT",
            "subtract",
            "NIQ",
            "maximum",
            "PT",
            "NIQ",
            "add",
            "OWT",
            "multiply",
            "multiply",
            "add",
            "PT",
            "multiply",
            "multiply",
            "add",
            "PT",
            "MWT",
            "multiply",
            "PT",
            "WIQ",
            "multiply",
            "multiply",
            "add",
            "PT",
            "OWT",
            "multiply",
            "PT",
            "PT",
            "WIQ",
            "PT",
            "multiply",
            "PT",
            "WIQ",
        ],
        display_mode="light",
        html_action="return",
    )
    with open("attention_viz.html", "w", encoding="utf-8") as f:
        f.write(html_obj.data)
