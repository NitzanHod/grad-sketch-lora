import wandb
import torch


def mem_status(optim, enable_galore, is8bit=True, report=True):
    """
    Calculate memory spent on optim states (Adam moments + Galore projection matrices)
    """

    def numel_of_state(state):
        return sum(
            [x.numel() for param_state in state.values() for x in param_state.values() if isinstance(x, torch.Tensor)])

    # no galore option
    true_total_size = numel_of_state(optim.state_dict()['state'])
    if not enable_galore:
        if report:
            wandb.config.update({'optim_states': {'true_total_size': true_total_size}})
        return

    def get_ids(dicts):
        ids_with_special_key = [d['params'] for d in dicts if d.get('proj_type')]
        ids_without_special_key = [d['params'] for d in dicts if not d.get('proj_type')]
        flatten = lambda l: sum(map(flatten, l), []) if isinstance(l, list) else [l]
        return flatten(ids_with_special_key), flatten(ids_without_special_key)

    def count_proj(e_proj):
        # counts the optim states of the Projector (e.g. just |P| or |P|+|Q|)
        proj_elements = e_proj['projector'].ortho_matrix if isinstance(e_proj['projector'].ortho_matrix, list) else [
            e_proj['projector'].ortho_matrix]
        return sum([x.numel() if x is not None else 0 for x in proj_elements])

    # non galore layers
    state1 = 'state1' if is8bit else 'exp_avg'
    state2 = 'state2' if is8bit else 'exp_avg_sq'
    galore_param_ids, non_galore_param_ids = get_ids(optim.state_dict()['param_groups'])
    if report:
        print('galore_param_ids', galore_param_ids)
        print('non_galore_param_ids', non_galore_param_ids)
    galore_state_dict = {k: v for k, v in optim.state_dict()['state'].items() if k in galore_param_ids}
    non_galore_state_dict = {k: v for k, v in optim.state_dict()['state'].items() if k in non_galore_param_ids}

    true_galore_size = numel_of_state(galore_state_dict)
    true_non_galore_size = numel_of_state(non_galore_state_dict)

    # non galore layers
    r_m_size = sum([v[state1].numel() if state1 in v.keys() else 0 for v in non_galore_state_dict.values()])
    r_v_size = sum([v[state2].numel() if state2 in v.keys() else 0 for v in non_galore_state_dict.values()])

    # galore layers
    # print('galore_state_dict', galore_state_dict)
    m_size = sum([v[state1].numel() if state1 in v.keys() else 0 for v in galore_state_dict.values()])
    v_size = sum([v[state2].numel() if state2 in v.keys() else 0 for v in galore_state_dict.values()])
    proj_size = sum([count_proj(v) for v in galore_state_dict.values()])

    regular_cost = r_m_size + r_v_size
    galore_cost = m_size + v_size + proj_size
    total = regular_cost + galore_cost

    results = {'optim_states': {
        'total': total,
        'total_galore': galore_cost,
        'total_non_galore': regular_cost,
        'pq': proj_size,

        'true_galore_size': true_galore_size,
        'true_regular_size': true_non_galore_size,
        'true_total_size': true_total_size,

    }}
    if report:
        wandb.config.update(results)
        print('Memory Cost:')
        print(f'Total (m+v+pq): {total}, True Total (all): {true_total_size}')
        print(f'Regular layers: m={r_m_size}, v={r_v_size}, total={regular_cost} [true:{true_non_galore_size}]')
        print(f'GaLore layers: m={m_size}, v={v_size}, pq={proj_size}, total={galore_cost} [true:{true_galore_size}]')
    return results
