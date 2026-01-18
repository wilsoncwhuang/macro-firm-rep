import numpy as np

def forward_default(data, max_horizon=60, CRI=False):
    # Add extra 0 to the right column, and leave t+1 ~ T forward probability
    shifted_default = np.pad(data[:, 1, 1:],
                             ((0, 0), (0, 1)),
                             'constant')
    if CRI:
        survive_pd = (1.0 - data[:, 1] - data[:, 2])
        survive_pd_cumprod = np.cumprod(survive_pd, axis=1)
        forward_pd = survive_pd_cumprod * shifted_default
        return forward_pd[:, :max_horizon]
    else:
        # survival prob * shifted_default (output the conditional probability given surviing in the last stage)
        forward_pd = (1 - data[:, 1] - data[:, 2]) * shifted_default
        return forward_pd[:, :max_horizon]

def calc_cumulative_default(data):
    return np.cumsum(data, axis=1)

def get_foward_probability(
    max_horizon=60,
    CRI=True,
    predictions=None,
):
    probs = []
    for i in range(max_horizon):
        probs.append(predictions[:,i,:])
        if i == 0:
            data = np.zeros((probs[0].shape[0], 3))

    data = np.stack([data] + probs, axis=2)
    
    return forward_default(data, max_horizon=max_horizon, CRI=CRI)

def get_cumulative_probability(
    forward_probability
):
    return calc_cumulative_default(forward_probability)