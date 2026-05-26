from types import SimpleNamespace

from rmu_layer_utils import get_rmu_layer_module


def make_layer_list():
    return [object(), object(), object()]


def make_qwen_base_model():
    layers = make_layer_list()
    model = SimpleNamespace(
        config=SimpleNamespace(architectures=["Qwen2_5_VLForConditionalGeneration"]),
        model=SimpleNamespace(language_model=SimpleNamespace(layers=layers)),
    )
    return model, layers


def make_qwen_peft_model():
    updated_layers = make_layer_list()
    frozen_layers = make_layer_list()
    model = SimpleNamespace(
        config=SimpleNamespace(architectures=["Qwen2_5_VLForConditionalGeneration"]),
        base_model=SimpleNamespace(
            model=SimpleNamespace(language_model=SimpleNamespace(layers=updated_layers))
        ),
        model=SimpleNamespace(language_model=SimpleNamespace(layers=frozen_layers)),
    )
    return model, updated_layers, frozen_layers


def make_llava_model():
    layers = make_layer_list()
    model = SimpleNamespace(
        config=SimpleNamespace(architectures=["LlavaForConditionalGeneration"]),
        language_model=SimpleNamespace(model=SimpleNamespace(layers=layers)),
    )
    return model, layers


def main():
    model, layers = make_qwen_base_model()
    assert get_rmu_layer_module(model, 1, "frozen") is layers[1]
    assert get_rmu_layer_module(model, 2, "updated") is layers[2]

    model, updated_layers, frozen_layers = make_qwen_peft_model()
    assert get_rmu_layer_module(model, 0, "updated") is updated_layers[0]
    assert get_rmu_layer_module(model, 1, "frozen") is frozen_layers[1]

    model, layers = make_llava_model()
    assert get_rmu_layer_module(model, 2, "updated") is layers[2]

    print("RMU layer resolution tests passed.")


if __name__ == "__main__":
    main()
