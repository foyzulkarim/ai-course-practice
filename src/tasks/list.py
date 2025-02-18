# from transformers import pipelines

# # List all supported tasks
# tasks = pipelines.SUPPORTED_TASKS

# # Print default models for each task
# for task, config in tasks.items():
#     print("-" * 50)
#     print(f"Task: {task}")
#     print(f"Default Config: {config['default']}") # default model name
#     print("-" * 50)

from transformers import pipelines

# List all supported tasks
tasks = pipelines.SUPPORTED_TASKS

# Print the header
print("{:<25} {:<50} {:<50}".format("Task", "pt_model", "tf_model"))
print("=" * 125)

# Iterate through tasks and print each row in the table.
for task, config in tasks.items():
    default_config = config.get("default", {})
    pt_model = ""
    tf_model = ""
    
    # If default_config is a dict.
    if isinstance(default_config, dict):
        # Case 1: It contains a direct "model" key.
        if "model" in default_config:
            model_info = default_config["model"]
            pt_model = model_info.get("pt") or "N/A"
            tf_model = model_info.get("tf") or "N/A"
            if isinstance(pt_model, tuple):
                pt_model = pt_model[0] or "N/A"
            if isinstance(tf_model, tuple):
                tf_model = tf_model[0] or "N/A"
        # Case 2: It is a dict of language pairs (e.g., for the translation task)
        elif all(isinstance(key, tuple) for key in default_config.keys()):
            pt_models = []
            tf_models = []
            for lang_pair, conf in default_config.items():
                model_info = conf.get("model", {})
                pt = model_info.get("pt") or "N/A"
                tf = model_info.get("tf") or "N/A"
                if isinstance(pt, tuple):
                    pt = pt[0] or "N/A"
                if isinstance(tf, tuple):
                    tf = tf[0] or "N/A"
                pt_models.append(f"{lang_pair}: {pt}")
                tf_models.append(f"{lang_pair}: {tf}")
            pt_model = "; ".join(pt_models)
            tf_model = "; ".join(tf_models)
        else:
            # Other dict formats just fallback.
            pt_model = default_config or "N/A"
            tf_model = default_config or "N/A"
    else:
        # If not a dict, use default_config directly.
        pt_model = default_config or "N/A"
        tf_model = default_config or "N/A"
    
    print("{:<25} {:<50} {:<50}".format(task, pt_model, tf_model))
