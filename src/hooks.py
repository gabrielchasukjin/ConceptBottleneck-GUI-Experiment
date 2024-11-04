activations = []
contributions = []

def hook_fn(module, input, output):
    activations.append(output.detach().cpu().numpy())
    contributions.append((output @ module.weight.T).detach().cpu().numpy())

def register_hooks(model):
    return model.fc_concepts.register_forward_hook(hook_fn)
