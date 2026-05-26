def resolve_attr_path(root, path):
    current = root
    for attr in path.split("."):
        if not hasattr(current, attr):
            return None
        current = getattr(current, attr)
    return current


def get_rmu_layer_module(model, layer_id, model_role):
    arch = model.config.architectures[0]

    if "Qwen" in arch:
        candidate_paths = []
        if model_role == "updated":
            candidate_paths.extend([
                "base_model.model.language_model.layers",
                "model.language_model.layers",
                "language_model.layers",
            ])
        else:
            candidate_paths.extend([
                "model.language_model.layers",
                "base_model.model.language_model.layers",
                "language_model.layers",
            ])

        for path in candidate_paths:
            layers = resolve_attr_path(model, path)
            if layers is not None:
                return layers[layer_id]
    elif "Llava" in arch or "Gemma" in arch:
        layers = resolve_attr_path(model, "language_model.model.layers")
        if layers is not None:
            return layers[layer_id]

    raise ValueError(f"Unsupported model architecture: {arch}")
