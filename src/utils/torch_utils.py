import torch


def save_model(model, config, save_path, should_print=False):
    save_dict = {'model': model.state_dict(), 'config': config}
    torch.save(save_dict, save_path)
    print('Model saved to: %s' % save_path)


def load_model(path, model, device):
    model_dict = torch.load(path)
    config = model_dict['config']

    model.load_state_dict(model_dict['model'])
    model.to(device)
    return model, config


def get_num_params(model):
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_params


def stack_2D_tensors(groups):
    bsize = sum([x.size(0) for x in groups])
    maxlen = max([x.size(1) for x in groups])

    output = torch.zeros(bsize, maxlen, device=groups[0].device, dtype=groups[0].dtype)

    offset = 0
    for x in groups:
        endpos = offset + x.size(0)
        output[offset:endpos, :x.size(1)] = x
        offset = endpos

    return output


def stack_3D_tensors(groups):
    bsize = sum([x.size(0) for x in groups])
    maxlen = max([x.size(1) for x in groups])
    hdim = groups[0].size(2)

    output = torch.zeros(bsize, maxlen, hdim, device=groups[0].device, dtype=groups[0].dtype)

    offset = 0
    for x in groups:
        endpos = offset + x.size(0)
        output[offset:endpos, :x.size(1)] = x
        offset = endpos

    return output